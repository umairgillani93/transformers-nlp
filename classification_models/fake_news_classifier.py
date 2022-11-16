# setup the imports
import os
import sys
import torch
import pandas as pd 
import numpy as np 
import torch.nn as nn
import transformers
from transformers import Trainer
from transformers import TrainingArguments
from sklearn.model_selection import train_test_split


class config:
    MAX_LEN = 64
    TRAIN_BATCH_SIZE = 8
    TEST_BATCH_SIZE = 4
    EPOCHS = 2
    TRAIN_PATH = os.getenv('HOME') + '/datasets/fake_news/train.csv'
    TEST_PATH = os.getenv('HOME') + '/datasets/fake_news/test.csv'
    TOKENIZER_PATH = os.getenv('HOME') + '/models/bert_model/tokenizer/'
    TOKENIZER = transformers.BertTokenizer.from_pretrained(TOKENIZER_PATH)
    MODEL_PATH = os.getenv('HOME') + 'models/bert_model/model/'
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# Preprocess the dataset
class pp():
    df = pd.read_csv(config.TRAIN_PATH)
    df.drop([col for col in df.columns if col not in \
            ['text', 'label']], axis=1, inplace=True)
     
    text = df.text
    label = df.label


# prepare the dataset 
#class Dataset:
#    def __init__(self, text, label):
#        self.text = text
#        self.label = label
#        self.tokenizer = config.TOKENIZER
#        self.max_len = config.MAX_LEN
#
#
#    def __len__(self):
#        return len(self.labels)
#
#    
#    def __getitem__(self, idx):
#        '''
#        returns the row at given index
#        '''
#        text = str(self.text[idx])
#        text = " ".join(text.split())
#        
#        # create inputs
#        inputs = self.tokenizer.encode_plus(
#                text,
#                None,
#                max_length=self.max_len,
#                truncation=True,
#                pad_to_max_length=True
#                )
#
#        ids = inputs['input_ids']
#        attention_mask = inputs['attention_mask']
#        token_type_ids = inputs['token_type_ids']
#        train_xtrain_x
#        return {
#                'ids': torch.tensor(ids, dtype=torch.long),
#                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
#                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
#                'labels': torch.tensor(self.label[idx], dtype=torch.float)
#                }



class DataModel(torch.utils.data.Dataset):
    def __init__(self, text, labels):
        self.text = text 
        self.labels = labels 
    
    def __getitem__(self, idx):
        pass

                
train_x, val_x = pp.text[: int(len(pp.text) * 0.8)], pp.text[(int(len(pp.text) * 0.8)) + 1:]
train_y, val_y = pp.label[: int(len(pp.label) * 0.8)], pp.label[(int(len(pp.label) * 0.8)) + 1:]

print(f'\ntrain_encodings: {train_x}')
print(f'\nval_encodings: {val_x}')

tkz_train = config.TOKENIZER(train_x, truncation=True, padding=True)
tkz_val = config.TOKENIZER(val_x, truncation=True, padding=True)

print(f'done')
