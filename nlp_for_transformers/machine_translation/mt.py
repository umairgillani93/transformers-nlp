# machine translations using huggingface.com models

import os
import pprint 
import torch
import pandas as pd
import numpy as np 
import warnings
from transformers import pipeline
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import RegexpTokenizer
warnings.filterwarnings('ignore')

print('\nImports successful')

eng2spa = {}
pp = pprint.PrettyPrinter(indent=4)

for line in open('spa-eng/spa.txt'):
  line = line.rstrip()
  eng, spa = line.split('\t')

  if eng not in eng2spa:
    eng2spa[eng] = []

  else:
    eng2spa[eng].append(spa)


#pp.pprint('\neng-2-spa: {}'.format(eng2spa))

# blue function works with tokens rather than actual words
# so we have to tokenize the works using RegexTokenizer
tokenizer = RegexpTokenizer(r'\w+')

# let's calculate the blue score
# blue score - Bilingual Evaluation Understudy is the score that closely resembles with Human assesment
print('\nBlue score is: {}'.format(sentence_bleu([['hi']], ['hi'])))

# Using SmoothingFuntion to cater the sentences which have larger lenghts 
# than defined N-grams
smoother = SmoothingFunction()

print(sentence_bleu(['hi'], 'hi', smoothing_function=smoother.method4))

print('\n')
print(sentence_bleu([['this', 'is', 'test', 'sentence']], 'this is test sentence'.split()))

# Let's pre-tokenizer our targets for later use
eng2spa_tokens = {}
for eng, spa_list in eng2spa.items():
  spa_list_tokens = []
  for text in spa_list:
    tokens = tokenizer.tokenize(text.lower())
    spa_list_tokens.append(tokens)

  eng2spa_tokens[eng] = spa_list_tokens

pp.pprint('\neng-2-spanish tokens: {}'.format(eng2spa_tokens))


# Sentencepiece is required by the translation models
# which needs to be "pip install sentencepiece"

print('\n >> Downloading model...')
translator = pipeline("translation",
                      model="Helsinki-NLP/opus-mt-en-es")

path = os.getenv('HOME') + '/models/translation_pretrained_weights/'

print(' >> Saving model to the path: {}'.format(path))
translator.save_pretrained(path)
print('\nModel saved')
