import torch
from transformers import BertTokenizer, BertForSequenceClassification
# from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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

class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name)

    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]

        return loss

# num_labels = 2
# model = BertForSequenceClassification(num_labels)
model = BERT()
dat = pd.read_csv('test/IMDB Dataset.csv')
print(dat.head)

X = dat['review']
y = dat['sentiment']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
X_train = X_train.values.tolist()
X_test = X_test.values.tolist()

y_train = pd.get_dummies(y_train).values.tolist()
y_test = pd.get_dummies(y_test).values.tolist()

max_seq_length = 256

class text_dataset(Dataset):
    def __init__(self,x_y_list, transform=None):
        
        self.x_y_list = x_y_list
        self.transform = transform
        
    def __getitem__(self,index):
        
        tokenized_review = tokenizer.tokenize(self.x_y_list[0][index])
        
        if len(tokenized_review) > max_seq_length:
            tokenized_review = tokenized_review[:max_seq_length]
            
        ids_review  = tokenizer.convert_tokens_to_ids(tokenized_review)

        padding = [0] * (max_seq_length - len(ids_review))
        
        ids_review += padding
        
        assert len(ids_review) == max_seq_length
        
        #print(ids_review)
        ids_review = torch.tensor(ids_review)
        
        sentiment = self.x_y_list[1][index] # color        
        list_of_labels = [torch.from_numpy(np.array(sentiment))]
        
        
        return ids_review, list_of_labels[0]
    
    def __len__(self):
        return len(self.x_y_list[0])


batch_size = 32

train_lists = [X_train, y_train]
test_lists = [X_test, y_test]

training_dataset = text_dataset(x_y_list = train_lists )

test_dataset = text_dataset(x_y_list = test_lists )

dataloaders_dict = {'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
                   'val':torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
                   }
dataset_sizes = {'train':len(train_lists[0]),
                'val':len(test_lists[0])}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = xm.xla_device()
print(device)


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


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    print('==> Starting Training')
    scaler = GradScaler()
    # import pdb;pdb.set_trace()
    for epoch in range(num_epochs):
        epoch_time = time.time()
        tracker = xm.RateTracker()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.train()  # Set model to training mode
        running_loss = 0.0
        sentiment_corrects = 0            
        # Iterate over data.
        for step, (inputs, sentiment) in enumerate(dataloaders_dict['train']):
            sentiment = torch.max(sentiment.float(), 1)[1]
            inputs = inputs.to(device) 
            sentiment = sentiment.to(device)
            optimizer.zero_grad()
            loss = model(inputs, sentiment)            
            loss.backward()
            optimizer.step()
            scheduler.step()
            tracker.add(inputs.size(0))
            # if step % 100 == 0:
            _train_update(device, step, loss, tracker, epoch, None)
        
        time_elapsed = time.time() - epoch_time
        print(f'Epoch complete in {time_elapsed // 60}m {time_elapsed % 60}s')

    return model


lrlast = .001
lrmain = .00001
optimizer_ft = optim.Adam(model.parameters(), lr = lrlast)
criterion = nn.CrossEntropyLoss()

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

model = model.to(device)
model_ft1 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=10)