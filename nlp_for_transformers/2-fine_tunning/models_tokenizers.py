import os
import pprint
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

# prettry printer for python
pp = pprint.PrettyPrinter(indent=4)

# define checkpoint
# remember! this should be same for both 'Tokenizer' and 'Model'
checkpoint = 'bert-base-uncased'
# path to save the tokenizer
path = os.getenv('HOME') + '/tokenizers/AutoTokenizer/'

# instantiate the tokenizer object
print('\n >> Loading Tokenizer..')
tokenizer = AutoTokenizer.from_pretrained(path)

#tokenizer.save_pretrained(path)
#print('\nTokenizer saved at path: {}'.format(path))

print('\nTokenizer: {}'.format(tokenizer))

# token_type_ids don't appear in distil-bert
# bear in mind that it returns a list of length 4 for these two words "hello word"
# this is because one token is [SEP] at the start of the sentence
# and the other one is [CSL] at the end of sentence
pp.pprint(tokenizer("hello world"))

print('\nText tokens: {}'.format(tokenizer.tokenize("hello world")))

# converts the text tokens to int ids
ids = tokenizer.convert_tokens_to_ids("hello world".split())

print('\ntoken ids: {}'.format(ids))

# convert ids back to tokens
# for [SEP] and [CLS] tokens
print('\n', tokenizer.convert_ids_to_tokens([101, 102]))

# encoding and decoding of text
text = 'sky is blue'
print('\nEncode: {}'.format(tokenizer.encode(text)))
print('\nDecode: {}'.format(tokenizer.decode(tokenizer.encode(text))))

# Model inputs
# REturns the dictionary of input_ids -> integer ids for each token
# token_type_ids and attention mask
model_inputs = tokenizer(text)
pp.pprint('Model inputs: {}'.format(model_inputs))

data = [
    'sky is blue',
    'NLP is my craze'
    ]


pp.pprint('Data tokenizer: {}'.format(tokenizer(data)))
pp.pprint('Decode Data tokenizer: {}'.format(tokenizer.decode(tokenizer(data)['input_ids'][0])))
