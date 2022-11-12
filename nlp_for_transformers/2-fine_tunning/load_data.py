import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# define tokenizer
tkz_path = os.getenv('HOME') + '/models/autmodel/tokenizer'
tokenizer = AutoTokenizer.from_pretrained(tkz_path)

def tokenizer_fn(ex):
    return tokenizer(ex['text'], padding='max_length', truncation=True)

dataset = load_dataset('csv',
        'yelp_train.csv')

tkz_dataset = dataset.map(tokenizer_fn,  batched=True)

print(f'tokenized dataaset: {tkz_dataset}')


