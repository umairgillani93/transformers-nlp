import os 
from transformers import DistilBertTokenizer, DistilBertModel

print('Imports done')

PATH = os.getenv('HOME') + '/models/distilbert/tokenizer/'

chkpt = 'distilbert-base-uncased'
tkz = DistilBertTokenizer.from_pretrained(chkpt)
print('saving tokenizer..')
tkz.save_pretrained(PATH)
print(f'saved to path: {PATH}')
