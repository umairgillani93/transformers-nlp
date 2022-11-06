import os
import torch
from transformers import DistilBertModel, DistilBertTokenizer
print('IMports  done')

PATH = os.getenv('HOME') + '/models/distilbert/model/'

chkpt = 'distilbert-base-uncased'
model = DistilBertModel.from_pretrained(chkpt)
print('saving model...')

model.save_pretrained(PATH)
print(f'model saved to path: {PATH}')

