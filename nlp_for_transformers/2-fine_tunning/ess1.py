import os
import torch
from transformers import DistilBertTokenizer, DistilBertModel

seq = 'this is a test sequence'

chkpt = 'distilbert-base-cased'
path = os.getenv('HOME') + '/models/distilbert/tokenizer'
tkz = DistilBertTokenizer.from_pretrained(chkpt)

tkz.save_pretrained(path)

print('Tokenizer saved')
