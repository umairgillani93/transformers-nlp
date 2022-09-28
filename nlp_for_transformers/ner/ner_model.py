import os
import pickle 
import numpy as np 
import pandas as pd
from transformers import pipeline 

print('\nImports successful')
path = os.getenv('HOME') + '/ner_pretrained_weights/'

print('\n >> Downloading model')
ner = pipeline('ner', path, aggregation_strategy='simple', device=0)

with open('ner_train.pkl', 'rb') as f:
  corpus_train = pickle.load(f)

with open('ner_test.pkl', 'rb') as f:
  corpus_test = pickle.load(f)
  
print('\ncorpus test: {}'.format(corpus_test))

inputs - []
targets - []

for stp in corpus_test:
  tokens = []
  targest = []
  for token, tag in stp:
    tokens.append(token)
    targest.append(tag)o
  inputs.append(tokens)
  targest.append(target)

