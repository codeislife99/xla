from torch import optim
from transformers import BertTokenizer, BertForMaskedLM
import torch
from tqdm import tqdm  # for our progress bar
from transformers import AdamW
import torch_xla.core.xla_model as xm
import torch_xla.test.test_utils as test_utils
import torch_xla.distributed.xla_multiprocessing as xmp
import time
import argparse
import os
# import tensorflow as tf
# policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
# tf.keras.mixed_precision.experimental.set_policy(policy)

def _train_update(device, step, loss, tracker, epoch, writer):
    test_utils.print_training_update(
        device,
        step,
        loss,  
        tracker.rate(),
        tracker.global_rate(),
        epoch,
        summary_writer=writer,
    )

class MeditationsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)


def loop_with_amp(model, input_ids, attention_mask, labels, optimizer, autocast, scaler):
    with autocast():
        outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss

    if xla_enabled:
        scaler.scale(loss).backward()
        gradients = xm._fetch_gradients(optimizer)
        xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
        scaler.step(optimizer)
        scaler.update()
        xm.mark_step()
    else:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return loss, optimizer

def loop_without_amp(model, input_ids, attention_mask, labels, optimizer):
    outputs = model(input_ids, attention_mask=attention_mask,
                    labels=labels)
    loss = outputs.loss
    loss.backward()
    if xla_enabled:
        xm.optimizer_step(optimizer)
    else:
        optimizer.step()
    return loss, optimizer

def get_dataset_loader():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    with open(dataset_path, 'r') as fp:
        text = fp.read().split('\n')

    print(text[:5])

    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

    inputs['labels'] = inputs.input_ids.detach().clone()
    print(f"Input Keys:{inputs.keys()}")

    rand = torch.rand(inputs.input_ids.shape)
    # create mask array
    mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * \
            (inputs.input_ids != 102) * (inputs.input_ids != 0)

    selection = []

    for i in range(inputs.input_ids.shape[0]):
        selection.append(
            torch.flatten(mask_arr[i].nonzero()).tolist()
        )

    for i in range(inputs.input_ids.shape[0]):
        inputs.input_ids[i, selection[i]] = 103

    dataset = MeditationsDataset(inputs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    return loader

def get_device():
    if xla_enabled:
        return xm.xla_device()
    else:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    
def get_model(device):
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.to(device)
    model.train()
    return model

def get_autocast_and_scaler():
    autocast, scaler = None, None
    if amp_enabled:
        if xla_enabled: 
            from torch_xla.amp import autocast, GradScaler
            return autocast, GradScaler()
    
        from torch.cuda.amp import autocast, GradScaler
        return autocast, GradScaler()

    return autocast, scaler

def train(loader, device, model, optimizer, autocast, scaler):
    for epoch in range(num_epochs):
        # setup loop with TQDM and dataloader
        # loop = tqdm(loader, leave=True)
        
        start_time = time.time()
        for step, batch in enumerate(loader):
            # initialize calculated gradients (from prev step)
            optimizer.zero_grad()
            # pull all tensor batches required for training
            tracker = xm.RateTracker()  # Placing the tracker here frees it of I/O time. 
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)            
            if not xla_enabled:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
        
            # if step == 0:
            #     trace_model(model, input_ids, attention_mask, labels)
            #     import sys
            #     sys.exit()
 
            # process
            if amp_enabled:
                loss, optimizer = loop_with_amp(model, input_ids, attention_mask, labels, optimizer, autocast, scaler)
            else:
                loss, optimizer = loop_without_amp(model, input_ids, attention_mask, labels, optimizer)

            tracker.add(input_ids.size(0))
        num_steps = step + 1
        end_time = time.time()
        print("Epoch ", epoch, (end_time - start_time)/num_steps)
        # loop.set_description(f'Epoch {epoch}')
        
        _train_update(device, step, loss, tracker, epoch, None)

        # print relevant info to progress bar
        # loop.set_description(f'Epoch {epoch}')
        # loop.set_postfix(loss=loss.item())

def get_hlo_dumps():
    os.environ['TF_CPP_MIN_LOG_LEVEL']='0'
    # os.environ['TF_CPP_VMODULE']="hlo_pass_pipeline"="1"
    os.environ['XLA_FLAGS']=f"--xla_dump_to=/pytorch/xla/test/bert_mlm_hlo_no_doubles --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=.*"


def trace_model(model, input_ids, attention_mask, labels):
    traced_model = torch.jit.trace(model, [input_ids, attention_mask, labels])
    torch._C._jit_pass_onnx_function_substitution(traced_model.graph)
    print(traced_model.graph)

def main():
    loader = get_dataset_loader()
    device = get_device()
    model = get_model(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    autocast, scaler = get_autocast_and_scaler()

    if dlprof_enabled and not xla_enabled:
        with torch.autograd.profiler.emit_nvtx():
            train(loader, device, model, optimizer, autocast, scaler)
    else:
        train(loader, device, model, optimizer, autocast, scaler)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--xla_enabled', action='store_true', default=False)
    arg_parser.add_argument('--dlprof_enabled', action='store_true', default=False)
    arg_parser.add_argument('--xla_debug_metrics_enabled', action='store_true', default=False)
    args = arg_parser.parse_args()

    amp_enabled = True
    dump_hlo_graphs = False
    xla_enabled = args.xla_enabled
    xla_debug_metrics_enabled = args.xla_debug_metrics_enabled
    dlprof_enabled = args.dlprof_enabled


    if dump_hlo_graphs:
        get_hlo_dumps()

    if dlprof_enabled and not xla_enabled:
        import nvidia_dlprof_pytorch_nvtx
        nvidia_dlprof_pytorch_nvtx.init()

    if xla_enabled: 
        dataset_path = "/pytorch/xla/data/clean.txt"
    else:
        dataset_path = "data/clean.txt"

    num_epochs = 10
    main()
    if xla_enabled and xla_debug_metrics_enabled:
        import torch_xla.debug.metrics as met
        print(met.metrics_report())