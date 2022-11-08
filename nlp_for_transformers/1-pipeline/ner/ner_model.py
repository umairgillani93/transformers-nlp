import os
import pickle 
import torch
import numpy as np 
import pandas as pd
from transformers import pipeline 
from nltk.tokenize.treebank import TreebankWordDetokenizer

import transformers

print('torch version: {}'.format(torch.__version__))
print('transformers version: {}'.format(transformers.__version__))
print('\nCuda Available: {}'.format(torch.cuda.is_available()))
print('\nImports successful')
path = os.getenv('HOME') + '/ner_pretrained_weights/'
print(f'\nPath: {path}')

print('\n >> Loading model')
ner = pipeline('ner', path, aggregation_strategy='simple')
#print('\nSaving model path: {}'.format(path))
#ner.save_pretrained(path)
print('\nModel saved')
#ner = pipeline('ner', path)
print('\nMOdel loaded!')
with open('ner_train.pkl', 'rb') as f:
  corpus_train = pickle.load(f)

with open('ner_test.pkl', 'rb') as f:
  corpus_test = pickle.load(f)
  
inputs = []
targets = []

for stp in corpus_test:
  tokens = []
  target = []
  for token, tag in stp:
    tokens.append(token)
    target.append(tag)
  inputs.append(tokens)
  targets.append(target)

print('\nInputs: {}'.format(inputs[9]))

# for joining / detokenizing the work tokens
detokenizer = TreebankWordDetokenizer()

print(detokenizer.detokenize(inputs[9]))
print('\ntargets: {}'.format(targets[9]))
print(ner(detokenizer.detokenize(inputs[9])))


