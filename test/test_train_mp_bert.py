import torch
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
import torch_xla.test.test_utils as test_utils
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp


def _train_update(device, step, loss, tracker, epoch, writer):
    test_utils.print_training_update(
        device,
        step,
        loss.item(),
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

def train_bert(dataset_path, xla_enabled, amp_enabled):
    max_seq_length = 256
    batch_size = 32
    num_epochs = 25
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model = BERT()
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
    if amp_enabled:
        autocast, scaler = get_autocast_and_scaler(xla_enabled)

    if xla_enabled:
        import torch_xla.distributed.parallel_loader as pl
        train_device_loader = pl.MpDeviceLoader(dataloaders_dict['train'], device)
    else:
        train_device_loader = dataloaders_dict['train']

    for epoch in range(num_epochs):
        epoch_time = time.time()
        # tracker = xm.RateTracker()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.train()  # Set model to training mode          
        # Iterate over data.
        for step, (inputs, sentiment) in enumerate(train_device_loader):
            # import pdb;pdb.set_trace()
            # sentiment = torch.max(sentiment.float(), 1)[1]
            tracker = xm.RateTracker()

            if not xla_enabled:
                inputs = inputs.to(device) 
                sentiment = sentiment.to(device)
            optimizer.zero_grad()
            if amp_enabled:
                loss, optimizer = loop_with_amp(model, inputs, sentiment, optimizer, xla_enabled, autocast, scaler)
            else:
                loss, optimizer = loop_without_amp(model, inputs, sentiment, optimizer, xla_enabled)
            tracker.add(inputs.size(0))
            _train_update(device, step, loss, tracker, epoch, None)
        
        time_elapsed = time.time() - epoch_time
        print(f'Epoch complete in {time_elapsed // 60}m {time_elapsed % 60}s')

def _mp_fn(index):
    torch.set_default_tensor_type("torch.FloatTensor")
    train_bert(dataset_path, xla_enabled, amp_enabled)

if __name__ == "__main__":
    xla_enabled = False
    amp_enabled = True

    if xla_enabled:
        dataset_path = '/pytorch/xla/test/IMDB Dataset.csv'
        xmp.spawn(_mp_fn, nprocs=1)
    else:
        dataset_path = "test/IMDB Dataset.csv"
        train_bert(dataset_path, xla_enabled, amp_enabled)
