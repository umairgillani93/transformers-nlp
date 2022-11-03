import os
import torch
import torchinfo
import pprint
import numpy as np 
import pandas as pd 
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments
from datasets import load_dataset
from datasets import load_metric
import warnings
warnings.filterwarnings('ignore')

print('Imports successful')
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

print('\nDevice: {}'.format(device))

raw_dataset = load_dataset('glue', 'rte')
print('\ntrain features: ', raw_dataset['train'].features)

print('\nRaw dataset: {}'.format(raw_dataset))

def save_model(path, model, chkpt):
  '''
  downloads and saves the model from Huggingface.co
  '''
  print('\n >> Downloading model ..')
  model_ = model.from_pretrained(chkpt)
  print('\n >> Saving model to path: {}'.format(path))
  model_.save_pretrained(path)
  print('\nMOdel saved!')



# prettry printer for python
pp = pprint.PrettyPrinter(indent=4)

# define checkpoint
# remember! this should be same for both 'Tokenizer' and 'Model'
checkpoint = 'distilbert-base-uncased'
# path to save the tokenizer
tokenizer_path = os.getenv('HOME') + '/tokenizers/AutoTokenizer/'
model_path = os.getenv('HOME') + '/models/AutoModelForSequenceClassification_distilbert_pretrained/'

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

print('\nModel & Tokenize loaded')

print('Tokenized sequences: {}'.format(tokenizer(raw_dataset['train']['sentence1'][0],
                                                 raw_dataset['train']['sentence1'][0])))

pprint('\ntokenized sequences: {}'.format(tokenizer((raw_dataset['train']['sentence1'][0],
                                                    raw_dataset['train']['sentence2'][0]))))

# line added!
