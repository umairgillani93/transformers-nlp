import os 
from transformers import AutoTokenizer

# define checkpoint
# remember! this should be same for both 'Tokenizer' and 'Model'
checkpoint = 'bert-base-uncased'
# path to save the tokenizer
path = os.getenv('HOME') + '/tokenizers/AutoTokenizer/'

# instantiate the tokenizer object
print('\n >> Loading Tokenizer..')
tokenizer = AutoTokenizer.from_pretrained(checkpoint, path)

print('\n >> Tokenizer loaded!')
#tokenizer.save_pretrained(path)
#print('\nTokenizer saved at path: {}'.format(path))


