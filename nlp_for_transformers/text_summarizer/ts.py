import os 
import torch
import textwrap
import numpy as np 
import pandas as pd 
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

print('\nImports successful')
path = os.getenv('HOME') + '/summarizer_pretrained_weights/'

def print_summary(doc):
  '''prints the summary of passed document'''
  result = summarize(doc.iloc[0].split('\n', 1)[1])
  print(wrap(result[0]['summary_text']))
  
def wrap(x):
  '''returns the wraped text'''
  return textwrap.fill(x, replace_whitespace=False, fix_sentence_endings=True)

# load the data
df = pd.read_csv('bbc_text_cls.csv')

# choose randome article
doc = df[df['labels'] == 'business']['text'].sample(random_state=42)

print('\ndocument: {}'.format(doc))

# define summarizer pretrained model
print('\n >> Downloading model..')
summarizer = pipeline('summarization', path)

#print('\n >> Saving model.. ')
#summarizer.save_pretrained(path)
#print('\nModel saved to path: {}'.format(path))

summarizer(doc.iloc[0].split('\n', 1)[1])

print(print_summary(doc))


