from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def get_dataset():
    raw_datasets = load_dataset("imdb")
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
    # full_train_dataset = tokenized_datasets["train"]
    # full_eval_dataset = tokenized_datasets["test"]

    return small_train_dataset, small_eval_dataset

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")    
    small_train_dataset, small_eval_dataset = get_dataset()
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
    training_args = TrainingArguments('test_trainer', fp16=True, deepspeed='deepspeed_config.json',per_device_train_batch_size=8)
    # training_args = TrainingArguments('test_trainer', fp16=True, do_train=True)
    print(training_args)
    trainer= Trainer(model= model, args=training_args, train_dataset=small_train_dataset, eval_dataset=small_eval_dataset)
    trainer.train()

