import os 
import json 
import textwrap 
import numpy as np 
import matplotlib.pyplot as plt 
from pprint import pprint
import warnings
from transformers import pipeline 
from transformers import set_seed # for setting same values each time
warnings.filterwarnings('ignore')

print('>>imports successful')
PATH = os.getcwd() + '/dataset'

def load_data(path):
  '''
  returns the pre-processed data
  '''
  for filename in os.listdir(path):
    path_ = os.path.join(PATH, filename)
    print(f'file path: {path_}')

  with open(path_, 'r') as f:
    result = [x.strip() for x in f.read().split('\n') if len(x) > 0]

  return result

gen = pipeline('text-generation')

set_seed(1234)

data = load_data(PATH)

print(data[0])



