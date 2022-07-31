import os
import numpy as np
import pandas as pd
import transformers
import torch 
import torch.nn as nn 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModel, BertTokenizerFast
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

device = torch.device('cude') if torch.cuda.is_available() else 'cpu'

print('device: {}'.format(device))

df = pd.read_csv('./spam.csv')
# print(df.head())

train_x, test_x, train_y, test_y = train_test_split(
                                                  df['text'], df['label'],
                                                  test_size=0.3,
                                                  stratify=df['label']
                                                  )


val_text, test_x, val_labels, test_y= train_test_split(test_x, test_y,
                                                                random_state=2018,
                                                                test_size=0.5,
                                                                stratify=test_y)



# Load BERT-base pretrained model
model = AutoModel.from_pretrained('bert-base-uncased')

# Load BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# sample data
text = [
    'this is first line',
    'this is second list',
    'sky is blue',
    'fire is hot'
    ]

sentence_ids = tokenizer.batch_encode_plus(
                                          text,
                                          padding = True,
                                          return_token_type_ids = True)

print(f'\nSentence_ids: {sentence_ids}')
print(f'\nKeys: {sentence_ids.keys()}')

# Tokenization
seq_len = [len(x.split()) for x in train_x]
max_len = 25
pd.Series(seq_len).hist(bins=30)

# Tokenizer and encode sequnces in training
tokens_train = tokenizer.batch_encode_plus(
    train_x.tolist(),
    max_length=max_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False)

tokens_val= tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length=max_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False)

tokens_test = tokenizer.batch_encode_plus(
    test_x.tolist(),
    max_length=max_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False)

 # convert integer sequences to Tensors
 # for train set
train_seq = torch.Tensor(tokens_train['input_ids'])
train_mask = torch.Tensor(tokens_train['attention_mask'])
train_y = torch.Tensor(train_y.tolist())
print('done for trainings')
print(f'\ntrain_y: {train_y}')

# for validation set
val_seq = torch.Tensor(tokens_val['input_ids'])
val_mask = torch.Tensor(tokens_val['attention_mask'])
val_y = torch.Tensor(val_labels.tolist())

# for test set
test_seq = torch.Tensor(tokens_test['input_ids'])
test_mask = torch.Tensor(tokens_test['attention_mask'])
test_y = torch.Tensor(test_y.tolist())

# Data Loaders
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

batch_size = 32

# wrapping Tensors
train_data = TensorDataset(
                          train_seq,
                          train_mask,
                          train_y)

# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)

# data loader for train set
train_dataloader = DataLoader(
                            train_data,
                            sampler=train_sampler,
                            batch_size=batch_size)

val_data = TensorDataset(
                          val_seq,
                          val_mask,
                          val_y)

# sampler for sampling the data during training
val_sampler = RandomSampler(val_data)

# data loader for train set
val_dataloader = DataLoader(
                            val_data,
                            sampler=val_sampler,
                            batch_size=batch_size)


# Freeze BERT parameters
print('total paramters: {}'.format(len(list(model.parameters()))))

for param in model.parameters():
  param.requires_grad = False

# Define Custom head
class CustomHead(nn.Module):
  def __init__(self, model):
    super(CustomHead, self).__init__()
    self.model = model

    # dropout layer
    self.dropout = nn.Dropout(0.1)

    # rele activation
    self.relu = nn.ReLU()

    # first dense
    self.fc1 = nn.Linear(768, 512) # input -> 758 (attention_head), output -> 512 
    # second dense
    self.fc1 = nn.Linear(512, 2)

    # softmax activation function
    self.softmax = nn.LogSoftmax(dim=1)

    # forward pass
    def forward(self, sent_id, mask):
      # pass inputs to the model
      _, cls_hs, self.model(sent_id, attention_mask=mask)
      x = self.fc1(cls_hs)
      x = self.relu(x)
      x = self.dropout(x)

      # output layer
      x = self.fc2(x)

      # softmax activation
      x = self.softmax(x)
      return x


# pass the pre-trained BERT to out defined architecture
model_ = CustomHead(model)

# pushing model to GPU
model_ = model.to(device)

from transformers import AdamW
# define optimzer
optim = AdamW(model.parameters(), lr=1e-3)

#compute the class weights
print('unique labels: {}'.format(np.unique(train_y)))
class_wts = compute_class_weight('balanced', np.unique(train_y), train_y)
print('\nclass weights: {}'.format(class_wts))

