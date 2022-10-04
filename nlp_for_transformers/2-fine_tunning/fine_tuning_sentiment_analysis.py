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

raw_dataset = load_dataset('glue', 'sst2')

print('\nRaw dataset: {}'.format(raw_dataset))



# prettry printer for python
pp = pprint.PrettyPrinter(indent=4)

# define checkpoint
# remember! this should be same for both 'Tokenizer' and 'Model'
checkpoint = 'bert-base-uncased'
# path to save the tokenizer
tokenizer_path = os.getenv('HOME') + '/tokenizers/AutoTokenizer/'
model_path = os.getenv('HOME') + '/models/AutoModelForSequenceClassification_pretrained/'

# load the tokenizer and model
print('\n >> Loading model..')
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
print('Model Loaded')

def tokenizer_fn(batch):
  return tokenizer(batch['sentence'], truncation=True)

# tokenize our raw dataset
tokenized_dataset = raw_dataset.map(tokenizer_fn, batched=True)

# create training_args object
training_args = TrainingArguments(
    'my_trainer',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=1)

print('\Model type: {}'.format(type(model)))

print('\nMOdel summary: {}'.format(torchinfo.summary(model)))

# Since when we train a Transformer model we train all the weights of it
# So we are gonna save the weights / parameters before our training 
# In order to verify after training
params_before = []
for name, p in model.named_parameters():
  params_before.append(p.detach().numpy())

metric = load_metric('glue', 'sst2')

print('\nscore: {}'.format(metric.compute(predictions=[1,0,1], references=[1,0,0])))

trainer = Trainer(
    mdoel,
    training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    tokenizer=tokenizer
    )

print('\n >> Training model ..')
trainer.train()
print('Model trained')




