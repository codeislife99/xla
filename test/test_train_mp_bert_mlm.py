from torch import optim
from transformers import BertTokenizer, BertForMaskedLM
import torch
from tqdm import tqdm  # for our progress bar
from transformers import AdamW
import torch_xla.core.xla_model as xm
import torch_xla.test.test_utils as test_utils
import torch_xla.distributed.xla_multiprocessing as xmp
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

def main():
    loader = get_dataset_loader()
    device = get_device()
    model = get_model(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    autocast, scaler = get_autocast_and_scaler()

    for step, epoch in enumerate(range(num_epochs)):
        # setup loop with TQDM and dataloader
        # loop = tqdm(loader, leave=True)
        for batch in loader:
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
        
            # process
            if amp_enabled:
                loss, optimizer = loop_with_amp(model, input_ids, attention_mask, labels, optimizer, autocast, scaler)
            else:
                loss, optimizer = loop_without_amp(model, input_ids, attention_mask, labels, optimizer)

            tracker.add(input_ids.size(0))
            _train_update(device, step, loss, tracker, epoch, None)

            # print relevant info to progress bar
            # loop.set_description(f'Epoch {epoch}')
            # loop.set_postfix(loss=loss.item())

if __name__ == "__main__":
    xla_enabled = False
    amp_enabled = True

    if xla_enabled: 
        dataset_path = "/pytorch/xla/data/clean.txt"
    else:
        dataset_path = "data/clean.txt"

    num_epochs = 2
    main()