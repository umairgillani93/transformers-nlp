# import Tokenizer and Model
# define check points > This should be same for both model and tokenizer
# load tokenizer and model
# follow the remaining pipeline

import os
import sys
import torch
import torchinfo 
import pprint
import numpy as np 
import pandas as pd 
from transformers import pipeline 
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import AutoModelForSequenceClassification

from transformers import Trainer
from transformers import TrainingArguments
from datasets import load_dataset
from datasets import load_metric
import warnings

warnings.filterwarnings('ignore')

print('\nimports successful')

# define device info
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

print(f'\ndevice: {device}')

# load dataset from Huggingface.com, this dataset is based on views and their
# corresponding star ratings
def extract_data(path):
    ''''
    extracts the data as similar to 
    datasets load_data object
    '''
    df = pd.read_csv(path)
    cols = ['text', 'stars']
    df.drop([col for col in df.columns if col not in cols], axis=1, inplace=True)

    # 80% train and 20% test split
    train, test = df.iloc[0:int(len(df) * 0.8)], df.iloc[int(len(df) * 0.8 + 1):]

    train.to_csv(os.getenv('HOME') + '/datasets/yelp_train.csv')
    test.to_csv(os.getenv('HOME') + '/datasets/yelp_test.csv')

    print('All done')



def load_data():
    '''
    loads the data from specified train and test 
    csv data files
    '''
    data_files = {'train': '/home/umairgillani/datasets/yelp_train.csv',
                  'test': '/home/umairgillani/datasets/yelp_test.csv'
                  }
    
    return load_dataset(
            'csv', data_files = data_files
            )


extract_data(os.getenv('HOME') + '/datasets/yelp.csv')
dataset = load_data()
print(f'\ndataset: {dataset}')
print(sys.exit('fininshed'))

# STEP#2 Tokenizing the dataset
tkz_path = os.getenv('HOME') + '/models/distilbert/tokenizer/'
chkpt = 'bert-base-cased'
tokenizer = DistilBertTokenizer.from_pretrained(tkz_path)

# create a tokenizer function
def save_model(path, chkpt):
    '''
    saves the model to the path defined
    '''
    print(f' >> loading tokenizer...')
    tokenizer = DistilBertTokenizer.from_pretrained(chkpt)
    print(f' >> saving tokenizer...')
    tokenizer.save_pretrained(path)
    print(f' >> loading model ...' )
    model_ = DistilBertModel.from_pretrained(chkpt)
    print(f' >> saving model ...')
    model_.save_pretrained(path)
    print(' Alll done')


def tokenizer_fn(ex):
    '''
    BERT tokenizer returns us the following keys:
        - input_ids
        - token_type_ids
        - attention_mask
        -
    '''
    return tokenizer(ex['text'], padding='max_length',
            truncation=True)

tkz_train = tokenizer_fn(pd.DataFrame(train))
tkz_test = tokenizer_fn(pd.DataFrame(test))

print(f'\ntokenized train: {tkz_train}')
