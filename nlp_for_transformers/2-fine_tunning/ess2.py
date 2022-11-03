import os
import torch
from transformers import DistilBertModel, DistilBertTokenizer

print('Imports successful')

# load and save model
chkpt = 'distilbert-base-ucased'
path = os.getenv('HOME') + '/models/distilbert/model'
model = DistilBertModel.from_pretrained(chkpt)
model.save_pretrained(path)

print('model loaded')
