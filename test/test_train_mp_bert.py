import torch
from torch.autograd.function import traceable
from transformers import BertTokenizer, BertForSequenceClassification
# from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from random import randrange
import torch.nn.functional as F
import pandas as pd
from transformers.models.bert import configuration_bert 
import torch_xla.test.test_utils as test_utils
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import multiprocessing
import torch.profiler
import time 

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

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name)
        # import pdb;pdb.set_trace()
    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]

        return loss

class BERTdownsized(nn.Module):
    def __init__(self):
        super(BERTdownsized, self).__init__()

        options_name = "bert-base-uncased"
        from transformers import BertConfig
        configuration = BertConfig()
        configuration.num_hidden_layers=num_hidden_layers
        self.encoder = BertForSequenceClassification(configuration)
        # import pdb;pdb.set_trace()
        print(self.encoder)
    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]

        return loss

class BERTNoDropout(nn.Module):
    def __init__(self):
        super(BERTNoDropout, self).__init__()

        options_name = "bert-base-uncased"
        from transformers import BertConfig
        from modeling_bert import BertForSequenceClassification
        configuration = BertConfig()
        configuration.num_hidden_layers=num_hidden_layers
        self.encoder = BertForSequenceClassification(configuration)
        # import pdb;pdb.set_trace()
        print(self.encoder)
    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]
        return loss 
        
class text_dataset(Dataset):
    def __init__(self,x_y_list, max_seq_length, tokenizer, transform=None):
        
        self.x_y_list = x_y_list
        self.transform = transform
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

        self.ids_review_list = [] 
        self.list_of_labels = []
        # for index in range(len(self.x_y_list[0])):
        self.reduced_size = 1000
        for index in range(self.reduced_size):
            tokenized_review = self.tokenizer.tokenize(self.x_y_list[0][index])
        
            if len(tokenized_review) > self.max_seq_length:
                tokenized_review = tokenized_review[:self.max_seq_length]
            ids_review  = self.tokenizer.convert_tokens_to_ids(tokenized_review)
            padding = [0] * (self.max_seq_length - len(ids_review))
            ids_review += padding
            assert len(ids_review) == self.max_seq_length
            ids_review = torch.tensor(ids_review)
            sentiment = self.x_y_list[1][index] # color        
            list_of_labels = [torch.from_numpy(np.array(sentiment))]
            # return ids_review, list_of_labels[0]      
            self.ids_review_list.append(ids_review)
            # import pdb;pdb.set_trace()
            self.list_of_labels.append(torch.max(list_of_labels[0],0)[1])
              
    def __getitem__(self,index):
        return self.ids_review_list[index], self.list_of_labels[index]        
        # tokenized_review = self.tokenizer.tokenize(self.x_y_list[0][index])
        # if len(tokenized_review) > self.max_seq_length:
        #     tokenized_review = tokenized_review[:self.max_seq_length]
            
        # ids_review  = self.tokenizer.convert_tokens_to_ids(tokenized_review)
        # padding = [0] * (self.max_seq_length - len(ids_review))
        # ids_review += padding
        # assert len(ids_review) == self.max_seq_length
        # ids_review = torch.tensor(ids_review)
        # sentiment = self.x_y_list[1][index] # color        
        # list_of_labels = [torch.from_numpy(np.array(sentiment))]
        # return ids_review, list_of_labels[0]
    
    def __len__(self):
        # return len(self.x_y_list[0])
        return self.reduced_size

def get_autocast_and_scaler(xla_enabled): 
    if xla_enabled: 
        from torch_xla.amp import autocast, GradScaler
        return autocast, GradScaler()
    
    from torch.cuda.amp import autocast, GradScaler
    return autocast, GradScaler()

def loop_with_amp(model, inputs, sentiment, optimizer, xla_enabled, autocast, scaler):
    with autocast():
        loss = model(inputs, sentiment)

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

def loop_without_amp(model, inputs, sentiment, optimizer, xla_enabled):
    loss = model(inputs, sentiment)            
    loss.backward()
    if xla_enabled:
        xm.optimizer_step(optimizer)
    else:
        optimizer.step()
    return loss, optimizer

def step_pytorch_profile(prof): 
    if pytorch_profile:
        prof.step()

def publish_cpu_mem_usage(cpu_mem_usage):
    if cpu_mem_usage: 
        import resource
        print(f" CPU Usage Before: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")

def full_train_epoch(epoch, num_epochs, model, train_device_loader, device, optimizer, autocast, scaler, **kwargs):
    epoch_time = time.time()
    # tracker = xm.RateTracker()
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    model.train()  # Set model to training mode          
    # Iterate over data.
    publish_cpu_mem_usage(cpu_mem_usage)
    for step, (inputs, sentiment) in enumerate(train_device_loader):
        if xla_enabled:
            if xla_profile: 
                if step == kwargs['start_profile_after_step']:
                    training_started.set()
        elif pytorch_profile and (step + 1)== kwargs['break_after_steps']:
            break 
        tracker = xm.RateTracker()  # Placing the tracker here frees it of I/O time. 
        if not xla_enabled:  # This section is not necessary (but doesn't cause any performance problems) for XLA 
            inputs = inputs.to(device) 
            sentiment = sentiment.to(device)
        # if step == 1:
        #     trace_model(model, inputs, sentiment)
        #     import sys
        #     sys.exit()
        optimizer.zero_grad()
        if amp_enabled:
            loss, optimizer = loop_with_amp(model, inputs, sentiment, optimizer, xla_enabled, autocast, scaler)
        else:
            loss, optimizer = loop_without_amp(model, inputs, sentiment, optimizer, xla_enabled)
        tracker.add(inputs.size(0))
        # _train_update(device, step, loss, tracker, epoch, None)
        if pytorch_profile: 
            kwargs['profiler'].step()
    time_elapsed = time.time() - epoch_time
    print(f'Epoch complete in {time_elapsed // 60}m {time_elapsed % 60}s')     

def trace_model(model, inputs, sentiment):
    traced_model = torch.jit.trace(model, [inputs, sentiment])
    torch._C._jit_pass_onnx_function_substitution(traced_model.graph)
    print(traced_model.graph)

def train_bert(dataset_path, xla_enabled, amp_enabled):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # model = BERT()
    model = BERTdownsized()
    # model = BERTNoDropout()
    dat = pd.read_csv(dataset_path)
    print(dat.head)

    X = dat['review']
    y = dat['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    X_train = X_train.values.tolist()
    X_test = X_test.values.tolist()

    y_train = pd.get_dummies(y_train).values.tolist()
    y_test = pd.get_dummies(y_test).values.tolist()



    train_lists = [X_train, y_train]
    test_lists = [X_test, y_test]

    training_dataset = text_dataset(x_y_list = train_lists, max_seq_length = max_seq_length, tokenizer= tokenizer)

    test_dataset = text_dataset(x_y_list = test_lists, max_seq_length = max_seq_length, tokenizer=tokenizer)

    dataloaders_dict = {'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
                    'val':torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
                    }
    dataset_sizes = {'train':len(train_lists[0]),
                    'val':len(test_lists[0])}

    if xla_enabled:
        device = xm.xla_device()
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)
    lrlast = 1e-3
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr = lrlast)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    print('==> Starting Training')
    autocast, scaler = None, None
    if amp_enabled:
        autocast, scaler = get_autocast_and_scaler(xla_enabled)
    
    if xla_enabled:
        import torch_xla.distributed.parallel_loader as pl
        if xla_profile:
            server = xp.start_server(port_number)
        train_device_loader = pl.MpDeviceLoader(dataloaders_dict['train'], device)
        # train_device_loader = dataloaders_dict['train']
    else:
        train_device_loader = dataloaders_dict['train']
    

    start_time = time.time()
    if dlprof_enabled: 
        if xla_enabled: # Profiling DLProf & XLA Profiler with XLA 
            for epoch in range(num_epochs):
                full_train_epoch(epoch, num_epochs, model, train_device_loader, device, optimizer, autocast, scaler, start_profile_after_step=5)
        else: # Profiling DLProf with PyTorch 
            with torch.autograd.profiler.emit_nvtx():
                for epoch in range(num_epochs):
                    full_train_epoch(epoch, num_epochs, model, train_device_loader, device, optimizer, autocast, scaler)
    elif pytorch_profile: # Using Default PyTorch Profiler
        with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=2, warmup=3, active=15),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(pytorch_tb_folder),
                record_shapes=True
        ) as prof:    
            for epoch in range(num_epochs):
                full_train_epoch(epoch, num_epochs, model, train_device_loader, device, optimizer, autocast, scaler, profiler=prof, break_after_steps=20)
    else: # Using XLA Profiler with XLA 
        for epoch in range(num_epochs):
            full_train_epoch(epoch, num_epochs, model, train_device_loader, device, optimizer, autocast, scaler, start_profile_after_step=5)
    total_epoch_time = time.time() - start_time 
    print(f"Epoch Time(s) for {num_epochs} epochs = {total_epoch_time}")

    if xla_enabled and xla_debug_metrics_enabled:
        import torch_xla.debug.metrics as met
        print(met.metrics_report())

def _mp_fn(index):
    torch.set_default_tensor_type("torch.FloatTensor")
    train_bert(dataset_path, xla_enabled, amp_enabled)

def download_dataset():
    dataset_dir = os.path.dirname(dataset_path) 
    os.system(f"wget -N -P {dataset_dir} https://raw.githubusercontent.com/sugi-chan/custom_bert_pipeline/master/IMDB%20Dataset.csv")

def get_hlo_dumps():
    os.environ['TF_CPP_MIN_LOG_LEVEL']='0'
    # os.environ['TF_CPP_VMODULE']="hlo_pass_pipeline"="1"
    os.environ['NUM_HIDDEN_LAYERS']=str(num_hidden_layers)
    os.environ['XLA_FLAGS']=f"--xla_dump_to=/pytorch/xla/test/bert_hlo_{num_hidden_layers} --xla_dump_hlo_as_text --xla_dump_hlo_as_dot --xla_dump_hlo_pass_re=.*"

if __name__ == "__main__":
    amp_enabled = True

    # PT-ony - disabel XLA
    xla_enabled = True
    xla_profile = False
    xla_debug_metrics_enabled = True
    xla_tb_folder = "/pytorch/xla/test/bert_xla_tensorboard"
    xla_dataset_path = '/pytorch/xla/test/IMDB Dataset.csv'

    dlprof_enabled = False
    cpu_mem_usage = False
    hlo_dump = False

    pt_dataset_path = "IMDB Dataset.csv"
    pytorch_profile = False
    pytorch_tb_folder = "test/bert_pt_tensorboard_non_amp_one_hidden_layer"
    
    num_hidden_layers = 12
    max_seq_length = 128
    batch_size = 16
    num_epochs = 10

    if hlo_dump:
        get_hlo_dumps()

    if dlprof_enabled and not xla_enabled: 
        import nvidia_dlprof_pytorch_nvtx
        nvidia_dlprof_pytorch_nvtx.init()
    if xla_enabled:
        import torch_xla.debug.profiler as xp
        port_number = 8192
        training_started = multiprocessing.Event()
        dataset_path = xla_dataset_path
        download_dataset()
        def target_fn():
            xmp.spawn(_mp_fn, nprocs=1)
        p = multiprocessing.Process(target=target_fn, args=())
        p.start()
        if xla_profile:
            training_started.wait()
            xp.trace(f'localhost:{port_number}', xla_tb_folder)
        # xmp.spawn(_mp_fn, nprocs=1)
    else:
        dataset_path = os.path.join(os.getcwd(), "IMDB Dataset.csv")
        download_dataset()
        train_bert(dataset_path, xla_enabled, amp_enabled)
