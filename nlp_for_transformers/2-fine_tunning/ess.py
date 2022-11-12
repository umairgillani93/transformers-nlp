# import Tokenizer and Model
# define check points > This should be same for both model and tokenizer
# load tokenizer and model
# follow the remaining pipeline

import os
import sys
import torch
#import evaluate
import torchinfo 
import pprint
import numpy as np 
import pandas as pd 
from transformers import pipeline 
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import AutoModelForSequenceClassification
from transformers import AutoModel, AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments
from datasets import load_dataset
from datasets import load_metric
import warnings

warnings.filterwarnings('ignore')

print('\nimports successful')

# define device info
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

print(f'\ndevice: {device}')

# load dataset from Huggingface.com, this dataset is based on views and their
# corresponding star ratings
def extract_data(path):
    ''''
    extracts the data as similar to 
    datasets load_data object
    '''
    df = pd.read_csv(path)
    cols = ['text', 'stars']
    df.drop([col for col in df.columns if col not in cols], axis=1, inplace=True)

    # 80% train and 20% test split
    train, test = df.iloc[0:int(len(df) * 0.8)], df.iloc[int(len(df) * 0.8 + 1):]

    train.to_csv(os.getenv('HOME') + '/datasets/yelp_train.csv')
    test.to_csv(os.getenv('HOME') + '/datasets/yelp_test.csv')

    print('All done')



def load_data():
    '''
    loads the data from specified train and test 
    csv data files
    '''
    data_files = {'train': '/home/umairgillani/datasets/yelp_train.csv',
                  'test': '/home/umairgillani/datasets/yelp_test.csv'
                  }
    
    return load_dataset(
            'csv', data_files = data_files
            )


#extract_data(os.getenv('HOME') + '/datasets/yelp.csv')
dataset = load_data()
print(f'\ndataset: {dataset}')

# STEP#2 Tokenizing the dataset
tkz_path = os.getenv('HOME') + '/models/automodel/tokenizer/'
model_path = os.getenv('HOME') + '/models/automodel/model/' 
chkpt = 'bert-base-cased'
tokenizer = AutoModel.from_pretrained(tkz_path)
model = AutoModel.from_pretrained(model_path)


print('Model loaded..')


# create a tokenizer function
def save_model(path, chkpt):
    '''
    saves the model to the path defined
    '''
    print(f' >> loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(chkpt)
    print(f' >> saving tokenizer...')
    tokenizer.save_pretrained(path)
    print(f' >> loading model ...' )
    model_ = AutoModel.from_pretrained(chkpt)
    print(f' >> saving model ...')
    model_.save_pretrained(path)
    print(' Alll done')


def tokenizer_fn(ex):
    '''
    BERT tokenizer returns us the following keys:
        - input_ids
        - token_type_ids
        - attention_mask
        -
    '''
    return tokenizer(ex['text'], padding='max_length',
            truncation=True)

tkz_train = tokenizer_fn(pd.DataFrame(train))
tkz_test = tokenizer_fn(pd.DataFrame(test))

print(f'\ntokenized train: {tkz_train}')

print(train.head())
print(test.head())

# define the model -> sequence classifiatoin BERT pretrained
model = AutoModel.from_pretraind(model_path, num_labels=5)
print('Model loaded..\n')


# define evaluation metric
#metric = evaluate.load('accuracy') # calculates the accuracy of predictions

def compute_metrics(eval_pred):
    '''
    computes the accuracy metrics
    by converting predictions to logits first
    '''
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions = predictions, references=labels)



# Setup the evaluation strategy to check the training progress
# After each epoch
# training hyperparameters
training_args = TrainingArguments(
        output_dir=os.getenv('HOME') + '/checkpoints/automodel/yelp_chkpt/',  # we can change this directly later
        evaluation_strategy='epoch'
        )

# Create a Trainer object
trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tkz_train,
        eval_dataset=tkz_test,
        compute_metrics=compute_metrics
        )

# Start fine-tuning the model
print(' >> Model training..\n')
trainer.train()


