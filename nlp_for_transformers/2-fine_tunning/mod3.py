import os
import torch
from transformers import DistilBertTokenizer

tok_path = os.getenv('HOME') + '/models/distilbert/tokenizer/'
tokenizer = DistilBertTokenizer.from_pretrained(tok_path)

print('loaded')
