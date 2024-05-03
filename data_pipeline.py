from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def to_sentiment(rating):
    '''
    classify sentiment into 3 categories
    '''
    rating = int(rating)
    if rating <= 2:
        return 0 # negative review
    elif rating == 3:
        return 1 # neutral review
    else:
        return 2 # positive review

def data_pipeline(tokenizer, csv_file, BATCH_SIZE, MAX_LEN):
    df = csv_file
    
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED, shuffle=True)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED, shuffle=True)

    train_text, train_labels = df_train.content.to_numpy().tolist(), df_train.sentiment.to_numpy().tolist()
    val_text, val_labels = df_val.content.to_numpy().tolist(), df_val.sentiment.to_numpy().tolist()
    test_text, test_labels = df_test.content.to_numpy().tolist(), df_test.sentiment.to_numpy().tolist()

    
    train_encodings = tokenizer(train_text, truncation=True, padding="max_length", max_length=MAX_LEN)
    val_encodings = tokenizer(val_text, truncation=True, padding="max_length", max_length=MAX_LEN)
    test_encodings = tokenizer(test_text, truncation=True, padding="max_length", max_length=MAX_LEN)
    
    train_dataset = ReviewDataset(train_encodings, train_labels)
    val_dataset = ReviewDataset(val_encodings, val_labels)
    test_dataset = ReviewDataset(test_encodings, test_labels)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(val_dataset, batch_size=8)
    test_dataloader = DataLoader(test_dataset)
    
    return train_dataloader, val_dataloader, test_dataloader