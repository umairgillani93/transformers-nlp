import os
import sys
import torch
from transformers import DistilBertModel

print('imports successful')

chkpt = 'distilbert-base-uncased'

model = DistilBertModel.from_pretrained(chkpt)

model_path = os.getenv('HOME') + '/models/distilbert/model/'
model.save_pretrained(model_path)

print('done')


