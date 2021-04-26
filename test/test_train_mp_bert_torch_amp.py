import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer
# from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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
from torch_xla.amp import autocast, GradScaler
import torch_xla.core.xla_model as xm

class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name)

    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]

        return loss

# class BertForSequenceClassification(nn.Module):
#     """BERT model for classification.
#     This module is composed of the BERT model with a linear layer on top of
#     the pooled output.
#     Params:
#         `config`: a BertConfig class instance with the configuration to build a new model.
#         `num_labels`: the number of classes for the classifier. Default = 2.
#     Inputs:
#         `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
#             with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
#             `extract_features.py`, `run_classifier.py` and `run_squad.py`)
#         `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
#             types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
#             a `sentence B` token (see BERT paper for more details).
#         `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
#             selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
#             input sequence length in the current batch. It's the mask that we typically use for attention when
#             a batch has varying length sentences.
#         `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
#             with indices selected in [0, ..., num_labels].
#     Outputs:
#         if `labels` is not `None`:
#             Outputs the CrossEntropy classification loss of the output with the labels.
#         if `labels` is `None`:
#             Outputs the classification logits of shape [batch_size, num_labels].
#     Example usage:
#     ```python
#     # Already been converted into WordPiece token ids
#     input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
#     input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
#     token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
#     config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
#         num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
#     num_labels = 2
#     model = BertForSequenceClassification(config, num_labels)
#     logits = model(input_ids, token_type_ids, input_mask)
#     ```
#     """
#     def __init__(self, num_labels=2):
#         super(BertForSequenceClassification, self).__init__()
#         self.num_labels = num_labels
#         self.bert = BertModel.from_pretrained('bert-base-uncased')
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, num_labels)
#         nn.init.xavier_normal_(self.classifier.weight)

#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
#         _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)

#         return logits

#     def freeze_bert_encoder(self):
#         for param in self.bert.parameters():
#             param.requires_grad = False
    
#     def unfreeze_bert_encoder(self):
#         for param in self.bert.parameters():
#             param.requires_grad = True

# from pytorch_pretrained_bert import BertConfig

# config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
#         num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

# num_labels = 2
# model = BertForSequenceClassification(num_labels)
model = BERT()
# Convert inputs to PyTorch tensors
text ='what is a pug'
zz = tokenizer.tokenize(text)
tokens_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(zz)])

# logits = model(tokens_tensor)
dat = pd.read_csv('/pytorch/xla/test/IMDB Dataset.csv')
print(dat.head)
zz = tokenizer.tokenize(dat.review[1])
z1z = tokenizer.tokenize(dat.review[3])
zzz = tokenizer.convert_tokens_to_ids(zz)
zzzz = tokenizer.convert_tokens_to_ids(z1z)
tokens_tensor = torch.tensor([zzz,zzz])
# logits = model(tokens_tensor)

from sklearn.model_selection import train_test_split
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


batch_size = 16

train_lists = [X_train, y_train]
test_lists = [X_test, y_test]

training_dataset = text_dataset(x_y_list = train_lists )

test_dataset = text_dataset(x_y_list = test_lists )

dataloaders_dict = {'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
                   'val':torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
                   }
dataset_sizes = {'train':len(train_lists[0]),
                'val':len(test_lists[0])}

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = xm.xla_device()
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
            inputs = inputs.to(device) 
            sentiment = sentiment.to(device)
            optimizer.zero_grad()
            with autocast(False):
                loss = model(inputs, sentiment.type(torch.LongTensor))
                # loss = criterion(outputs, torch.max(sentiment.float(), 1)[1])
            
            loss.backward()
            xm.optimizer_step(optimizer)
            scheduler.step()

            # scaler.scale(loss).backward()
            # gradients = xm._fetch_gradients(optimizer)
            # xm.all_reduce("sum", gradients, scale=1.0)
            # scaler.step(optimizer)
            # scaler.update()
            # xm.mark_step()
            # if scheduler:
            #     scheduler.step()

            tracker.add(inputs.size(0))

            # optimizer.step()
            # xm.optimizer_step(optimizer)
            # statistics
            # running_loss += loss.item() * inputs.size(0)
            sentiment_corrects += torch.sum(torch.max(outputs, 1)[1] == torch.max(sentiment, 1)[1])
            if step % 100 == 0:
                _train_update(device, step, loss, tracker, epoch, None)
        
        # epoch_loss = running_loss / dataset_sizes[phase]
        sentiment_acc = sentiment_corrects.double() / dataset_sizes['train']

        # print('{} total loss: {:.4f} '.format(phase,epoch_loss ))
        print(f'Sentiment_acc: {sentiment_acc}')
        time_elapsed = time.time() - epoch_time
        print(f'Epoch complete in {time_elapsed // 60}m {time_elapsed % 60}s')

    return model


lrlast = .001
lrmain = .00001
# optimizer_ft = optim.Adam(
#     [
#         {"params":model.bert.parameters(),"lr": lrmain},
#         {"params":model.classifier.parameters(), "lr": lrlast},
       
#    ])
optimizer_ft = optim.Adam(model.parameters(), lr = lrlast)
criterion = nn.CrossEntropyLoss()

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)
# if xm.is_master_ordinal():
#     writer = test_utils.get_summary_writer(FLAGS.logdir)
# lr_scheduler = schedulers.wrap_optimizer_with_scheduler(
#         optimizer,
#         scheduler_type="WarmupAndExponentialDecayScheduler",
#         scheduler_divisor=5,
#         scheduler_divide_every_n_epochs=20,
#         num_steps_per_epoch=num_training_steps_per_epoch,
#         summary_writer=None,
#     )

model = model.to(device)
model_ft1 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=10)
# import torch
# from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# # OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
# import logging
# logging.basicConfig(level=logging.INFO)

# # Load pre-trained model tokenizer (vocabulary)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
# import numpy as np
# import torchvision
# from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
# import time
# import os
# import copy
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# from random import randrange
# import torch.nn.functional as F
# import pandas as pd 
# import torch_xla.test.test_utils as test_utils
# from torch.cuda.amp import GradScaler, autocast
# import torch_xla.core.xla_model as xm

# class BertForSequenceClassification(nn.Module):
#     """BERT model for classification.
#     This module is composed of the BERT model with a linear layer on top of
#     the pooled output.
#     Params:
#         `config`: a BertConfig class instance with the configuration to build a new model.
#         `num_labels`: the number of classes for the classifier. Default = 2.
#     Inputs:
#         `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
#             with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
#             `extract_features.py`, `run_classifier.py` and `run_squad.py`)
#         `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
#             types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
#             a `sentence B` token (see BERT paper for more details).
#         `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
#             selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
#             input sequence length in the current batch. It's the mask that we typically use for attention when
#             a batch has varying length sentences.
#         `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
#             with indices selected in [0, ..., num_labels].
#     Outputs:
#         if `labels` is not `None`:
#             Outputs the CrossEntropy classification loss of the output with the labels.
#         if `labels` is `None`:
#             Outputs the classification logits of shape [batch_size, num_labels].
#     Example usage:
#     ```python
#     # Already been converted into WordPiece token ids
#     input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
#     input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
#     token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
#     config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
#         num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
#     num_labels = 2
#     model = BertForSequenceClassification(config, num_labels)
#     logits = model(input_ids, token_type_ids, input_mask)
#     ```
#     """
#     def __init__(self, num_labels=2):
#         super(BertForSequenceClassification, self).__init__()
#         self.num_labels = num_labels
#         self.bert = BertModel.from_pretrained('bert-base-uncased')
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, num_labels)
#         nn.init.xavier_normal_(self.classifier.weight)
#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
#         _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)

#         return logits
#     def freeze_bert_encoder(self):
#         for param in self.bert.parameters():
#             param.requires_grad = False
    
#     def unfreeze_bert_encoder(self):
#         for param in self.bert.parameters():
#             param.requires_grad = True

# from pytorch_pretrained_bert import BertConfig

# config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
#         num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

# num_labels = 2
# model = BertForSequenceClassification(num_labels)

# # Convert inputs to PyTorch tensors
# text ='what is a pug'
# zz = tokenizer.tokenize(text)
# tokens_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(zz)])

# logits = model(tokens_tensor)
# dat = pd.read_csv('test/IMDB Dataset.csv')
# print(dat.head)
# zz = tokenizer.tokenize(dat.review[1])
# z1z = tokenizer.tokenize(dat.review[3])
# zzz = tokenizer.convert_tokens_to_ids(zz)
# zzzz = tokenizer.convert_tokens_to_ids(z1z)
# tokens_tensor = torch.tensor([zzz,zzz])
# logits = model(tokens_tensor)

# from sklearn.model_selection import train_test_split
# X = dat['review']
# y = dat['sentiment']


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
# X_train = X_train.values.tolist()
# X_test = X_test.values.tolist()

# y_train = pd.get_dummies(y_train).values.tolist()
# y_test = pd.get_dummies(y_test).values.tolist()

# max_seq_length = 256

# class text_dataset(Dataset):
#     def __init__(self,x_y_list, transform=None):
        
#         self.x_y_list = x_y_list
#         self.transform = transform
        
#     def __getitem__(self,index):
        
#         tokenized_review = tokenizer.tokenize(self.x_y_list[0][index])
        
#         if len(tokenized_review) > max_seq_length:
#             tokenized_review = tokenized_review[:max_seq_length]
            
#         ids_review  = tokenizer.convert_tokens_to_ids(tokenized_review)

#         padding = [0] * (max_seq_length - len(ids_review))
        
#         ids_review += padding
        
#         assert len(ids_review) == max_seq_length
        
#         #print(ids_review)
#         ids_review = torch.tensor(ids_review)
        
#         sentiment = self.x_y_list[1][index] # color        
#         list_of_labels = [torch.from_numpy(np.array(sentiment))]
        
        
#         return ids_review, list_of_labels[0]
    
#     def __len__(self):
#         return len(self.x_y_list[0])


# batch_size = 16

# train_lists = [X_train, y_train]
# test_lists = [X_test, y_test]

# training_dataset = text_dataset(x_y_list = train_lists )

# test_dataset = text_dataset(x_y_list = test_lists )

# dataloaders_dict = {'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
#                    'val':torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
#                    }
# dataset_sizes = {'train':len(train_lists[0]),
#                 'val':len(test_lists[0])}

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)


# def _train_update(device, step, loss, tracker, epoch, writer):
#     test_utils.print_training_update(
#         device,
#         step,
#         loss.item(),
#         tracker.rate(),
#         tracker.global_rate(),
#         epoch,
#         summary_writer=writer,
#     )


# def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
#     print('==> Starting Training')
#     scaler = GradScaler()
#     for epoch in range(num_epochs):
#         epoch_time = time.time()
#         tracker = xm.RateTracker()
#         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         print('-' * 10)
#         model.train()  # Set model to training mode
#         running_loss = 0.0
#         sentiment_corrects = 0            
#         # Iterate over data.
#         for step, (inputs, sentiment) in enumerate(dataloaders_dict['train']):
#             inputs = inputs.to(device) 
#             sentiment = sentiment.to(device)
#             optimizer.zero_grad()
#             with autocast(False):
#                 outputs = model(inputs)
#                 outputs = F.softmax(outputs,dim=1)
#                 loss = criterion(outputs, torch.max(sentiment.float(), 1)[1])

#             loss.backward()
#             optimizer.step()
#             scheduler.step()
#             # scaler.scale(loss).backward()
#             # scaler.step(optimizer)
#             # scaler.update()
#             # scheduler.step()

#             tracker.add(inputs.size(0))

#             # optimizer.step()
#             # xm.optimizer_step(optimizer)
#             # statistics
#             # running_loss += loss.item() * inputs.size(0)
#             sentiment_corrects += torch.sum(torch.max(outputs, 1)[1] == torch.max(sentiment, 1)[1])
#             if step % 100 == 0:
#                 _train_update(device, step, loss, tracker, epoch, None)
        
#         # epoch_loss = running_loss / dataset_sizes[phase]
#         sentiment_acc = sentiment_corrects.double() / dataset_sizes['train']

#         # print('{} total loss: {:.4f} '.format(phase,epoch_loss ))
#         print(f'Sentiment_acc: {sentiment_acc}')
#         time_elapsed = time.time() - epoch_time
#         print(f'Epoch complete in {time_elapsed // 60}m {time_elapsed % 60}s')

#     return model


# lrlast = .001
# lrmain = .00001
# optim1 = optim.Adam(
#     [
#         {"params":model.bert.parameters(),"lr": lrmain},
#         {"params":model.classifier.parameters(), "lr": lrlast},
       
#    ])

# #optim1 = optim.Adam(model.parameters(), lr=0.001)#,momentum=.9)
# # Observe that all parameters are being optimized
# optimizer_ft = optim1
# criterion = nn.CrossEntropyLoss()

# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)
# model = model.to(device)
# model_ft1 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
#                        num_epochs=10)