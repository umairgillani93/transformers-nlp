import os 
import torch
from transformers import AutoTokenizer

print('Imports done')

PATH = os.getenv('HOME') + '/models/automodel/tokenizer/'

chkpt = 'bert-base-cased'
tkz = AutoTokenizer.from_pretrained(chkpt)
print('saving tokenizer..')
tkz.save_pretrained(PATH)
print(f'saved to path: {PATH}')
