import os
import sys
import torch
import transformers
from transformers import BertModel, BertTokenizer

PATH = os.getenv('HOME') + '/models/bert_model/tokenizer/'
tokenizer = BertTokenizer.from_pretrained(PATH)

data = [
        'this is first line',
        'this is second line'
        ]

print(tokenizer(data, truncation=True, padding=True))

