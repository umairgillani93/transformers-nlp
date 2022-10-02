import os
import torch
import textwrap
import pprint
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from transformers import pipeline 
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print('Imports successful')

path = os.getenv('HOME') + '/models/zero_shot_pretrained_weights/'
print('\n >> Loading model..')
# An advantage of zero-shot classification model is
# It can take any input and try to classify it
# Whether it's an image or a text sample
# So it's useful for labeling the image and text files as it doesn't require labels at the begining

classifier = pipeline('zero-shot-classification', path)
print('\nMOdel Loaded')
print('\n')
print(classifier("this is a great movie", candidate_labels=['positive', 'negative']))

text = """
  Due to the presence of isoforms of its components, there are 12 
  versions of AMPK in mammals
  """
print('\n')
print(classifier(text, candidate_labels=['biology', 'math', 'geology']))

# load the dataset
df = pd.read_csv('../text_summarizer/bbc_text_cls.csv')

print('\n', df.head())

# create a set of labels
labels = list(set(df['labels']))

print(textwrap.fill(df.iloc[1024]['text']))

preds = classifier(df['text'].tolist()[:10], candiate_labels=df['labels'][:10])

print('\nFinal predictions: {}'.format(preds))



