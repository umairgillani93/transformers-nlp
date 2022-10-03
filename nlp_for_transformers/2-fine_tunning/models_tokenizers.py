import os
import pprint
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
print('\nDevice: {}'.format(device))

def save_model(path, model, chkpt):
  '''
  downloads and saves the model from Huggingface.co
  '''
  print('\n >> Downloading model ..')
  model_ = model.from_pretrained(chkpt)
  print('\n >> Saving model to path: {}'.format(path))
  model_.save_pretrained(path)
  print('\nMOdel saved!')


# prettry printer for python
pp = pprint.PrettyPrinter(indent=4)

# define checkpoint
# remember! this should be same for both 'Tokenizer' and 'Model'
checkpoint = 'bert-base-uncased'
# path to save the tokenizer
tokenizer_path = os.getenv('HOME') + '/tokenizers/AutoTokenizer/'
model_path = os.getenv('HOME') + '/models/AutoModelForSequenceClassification_pretrained/'

# instantiate the tokenizer object
print('\n >> Loading Tokenizer..')
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

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
#model_inputs = tokenizer(text)
#pp.pprint('Model inputs: {}'.format(model_inputs))

data = [
    'test example number 1',
    'test exammple number 2'
    ]


pp.pprint('Data tokenizer: {}'.format(tokenizer(data)))
pp.pprint('Decode Data tokenizer: {}'.format(tokenizer.decode(tokenizer(data)['input_ids'][0])))


model_inputs = tokenizer(data, return_tensors='pt', truncation=True, padding=True) 
print('\nMOdel inputs', model_inputs)

#save_model(model_path, AutoModelForSequenceClassification, checkpoint)

model = AutoModelForSequenceClassification.from_pretrained(model_path, 
    num_labels=2)
    
print('\nSequence Model loaded successfully')

print('\nModel labels: {}'.format(model.num_labels))

# output of the model returns named_tuple / dictionary
# we can get each item by indexing, output['labels'] dictionary key grabbing way or simply by output.logis dot notation
outputs = model(**model_inputs)
print('\nModel outputs: {}'.format(outputs))

# remove the grad from logits in order to calculte the mean, probabilities etc
print('detached logits: {}'.format(outputs.logits.detach().numpy()))

model_inputs = tokenizer(data, return_tensors='pt', padding=True, truncation=True)
outputs = model(**model_inputs)

print('\nFinal model outputs: {}'.format(outputs))

