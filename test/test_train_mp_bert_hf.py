from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, BertTokenizer
from transformers import DistilBertForSequenceClassification, BertForSequenceClassification, Trainer, TrainingArguments, AdamW
import torch
import torch_xla.core.xla_model as xm
from torch.utils.data import DataLoader
import os

def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text(encoding='utf8'))
            labels.append(0 if label_dir is "neg" else 1)

    return texts, labels

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_bert(model_name, amp_enabled, dataset_path, num_examples=500):
    if model_name == "bert-base-uncased":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    elif model_name == 'distilbert-base-uncased':
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")


    train_texts, train_labels = read_imdb_split(os.path.join(dataset_path, 'train'))
    test_texts, test_labels = read_imdb_split(os.path.join(dataset_path,'test'))

    train_texts, train_labels = train_texts[:num_examples], train_labels[:num_examples]
    test_texts, test_labels = test_texts[:num_examples], test_labels[:num_examples]

    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)


    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = IMDbDataset(train_encodings, train_labels)
    val_dataset = IMDbDataset(val_encodings, val_labels)
    test_dataset = IMDbDataset(test_encodings, test_labels)

    device = xm.xla_device()
    model.to(device)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    optim = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(3):
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            xm.optimizer_step(optim)
            # optim.step()

    model.eval()

if __name__ == "__main__":
    dataset_path = "/pytorch/xla/test/aclImdb/"
    model_name = "bert-base-uncased"
    amp_enabled = False
    train_bert(model_name, amp_enabled, dataset_path)