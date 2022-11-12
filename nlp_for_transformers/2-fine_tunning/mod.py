import os
import torch
from transformers import AutoModel
print('IMports  done')

PATH = os.getenv('HOME') + '/models/autmodel/model/'

chkpt = 'bert-base-cased'
model = AutoModel.from_pretrained(chkpt)
print('saving model...')

model.save_pretrained(PATH)
print(f'model saved to path: {PATH}')

