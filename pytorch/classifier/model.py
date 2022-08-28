# Bert fine-tunning by UmairGillani; umairgillani93@gmail.com
# Importing the libraries
import os 
import time 
import numpy as np 
import pandas as pd
import torch 
import torch.nn as nn 
from sklearn.model_selection import train_test_split   
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 
import transformers
from transformers import BertModel, BertTokenizerFast # import BERT model and Tokenizer weights
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


print('Torch version: {}'.format(torch.__version__))
print('\nTransformers version: {}'.format(transformers.__version__))

# specify the device
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

print('\nDevice: {}'.format(device))

# load the dataframe
df = pd.read_csv('train.csv')

print(df.head())
print(df.shape)

# Let's split the dataset into 3 chunks 
# Train, Validation and Test

train_text, temp_text, train_labels, temp_labels = train_test_split(df['text'], df['labels'],
                                                                    random_state = 2018,
                                                                    test_size = 0.3)

# Let's now create a validation set from temp_text and temp_labels
val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                    random_state = 2018,
                                                                    test_size = 0.5)

print('\nValidation set defined')

# Define constants
# Our pretrained model will be loaded from this path, as we have already downloaded and 
# saved the model from Huggingface
path = os.getenv('HOME') + '/bert_pretrained_weights/'

# Load the Pre-trained model from defined path
print('\n>> Loading model')
bert = BertModel.from_pretrained(path)
print('>> Model Loaded from path: {}'.format(path))

# Load the tokenizer, this should be same as model weights
print('\nLoading tokenizer...')
tokenizer = BertTokenizerFast.from_pretrained(path)

# Lets tokenize a sample data to see how the tokenizer works
sample_text = [
    'this is first sentence',
    'this is a second sentence',
    'sky is blue',
    'fire is hot'
    ]


# Encode above sample_text defined
encoded_text = tokenizer.batch_encode_plus(sample_text,
                                           padding = True,
                                           return_token_type_ids = True)

print('\nEncoded text: {}'.format(encoded_text))
print('\nType encoded text: {}'.format(type(encoded_text)))
print(f'Encoded text keys: {encoded_text.keys()}')

# Plot the lengths of sequences in training dataset
sequence_len = [len(x.split()) for x in train_text]
pd.Series(sequence_len).hist(bins = 30)

# Define maximum sequeuence length
max_seq_len = 4

print('\nType trian text: {}'.format(type(train_text)))
# Tokenize and encode sequences in training data
train_tokens = tokenizer.batch_encode_plus(train_text.tolist(),
                                          max_length = max_seq_len,
                                          pad_to_max_length = True,
                                          truncation = True,
                                          return_token_type_ids = True)


# Tokenize and encode sequences in validation set
val_tokens = tokenizer.batch_encode_plus(val_text.tolist(),
                                           max_length = max_seq_len,
                                           pad_to_max_length = True,
                                           truncation = True,
                                           return_token_type_ids = True)


# Tokenizer and encode sequences in test set 
test_tokens = tokenizer.batch_encode_plus(test_text.tolist(),
                                           max_length = max_seq_len,
                                           pad_to_max_length = True,
                                           truncation = True,
                                           return_token_type_ids = True)


# Convert train sequences to tensors. 
# We'll only grab 'input_ids' and convert them to tensors
train_seq = torch.tensor(train_tokens['input_ids'])
# Convert attention_mask to tensors
train_mask = torch.tensor(train_tokens['attention_mask'])
# Convert train labels to tensors
train_y = torch.tensor(train_labels.tolist())

print('\nTrain seq: {}'.format(train_seq))
print('\nTrain mask: {}'.format(train_mask))
print('\nTrian y: {}'.format(train_y))

# Convert test sequences to tensors 
test_seq = torch.tensor(test_tokens['input_ids'])
# Convert attention_mask to tensors
test_mask = torch.tensor(test_tokens['attention_mask'])
# Convert labels to tensors
test_y = torch.tensor(test_labels.tolist())

print('\nTest seq: {}'.format(test_seq))
print('\nTest mask: {}'.format(test_mask))
print('\nTest y: {}'.format(test_y))

# convert valiation sequences to tensors
val_seq = torch.tensor(val_tokens['input_ids'])
# Convert validation attention_mask to tensor
val_mask =  torch.tensor(val_tokens['attention_mask'])
# Convert validation labels to tensor
val_y = torch.tensor(val_labels.tolist())

print('\nValid seq: {}'.format(val_seq))
print('\nValid mask: {}'.format(val_mask))
print('\nValid y: {}'.format(val_y))

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler

# Define batch size
batch_size = 2

# Wrap the Tensors inside TensorDataset
# Training data
train_data = TensorDataset(train_seq, train_mask, train_y)

# Sample the dataset during training 
train_sampler = RandomSampler(train_data)

# Data loader for training dataset
train_dataloader = DataLoader(train_data,
                              sampler = train_sampler,
                              batch_size = batch_size)

# Wrap the Tensor for validation dataset 
val_data = TensorDataset(val_seq, val_mask, val_y)

# Sample for sampling
val_sampler = SequentialSampler(val_data)

# DataLoader for validation dataset 
val_dataloader = DataLoader(val_data,
                            sampler = val_sampler,
                            batch_size = batch_size)


print('\n>>> Done smapling Training and Validation datasets')

# Freeze all the parameters of Pre-trained model
for p in bert.parameters():
  p.requires_grad = False

class CustomHead(nn.Module):
  '''
  mapped style dataset: Implement __getitem__ and __len__ protocols and returns index-wise mappings from dataset
  iterable style dataset: Implement __iter__ protocols and returns streams of data from remote servers / data sources and even logs
  '''
  def __init__(self, bert):
    # Define constructor function
    # Constructor function takes model as an input
    # Inherit base class member functions
    super(CustomHead, self).__init__()
    
    self.bert = bert

    # Add dropout layers -> this is not required in current data set {train.csv}. We'll change our dataset later
    self.dropout = nn.Dropout(0.001)

    # Add activation function 
    self.relu = nn.ReLU()

    # Add first dense layer
    self.fc1 = nn.Linear(768, 512)

    # Add second dense layer
    self.fc2 = nn.Linear(512, 2)

    # Add softmax activation function 
    self.softmax = nn.LogSoftmax(dim = 1)

    # Done with the Construtor function, this will get initialize and called when we instantiate our CustomHead class

  def forward(self, sent_ids, mask):
    ''' Forward-pass function takes two arguments
    sentence_ids or input_ids and attention mask, and runs
    forward-pass iteration on our dataset'''

    # Pass the inputs to the model
    _, pooled_opt = self.bert(send_ids, attention_mask = mask,
                              return_dict = False)


    # Pass Model's output to first dense layer
    x = self.fc1(pooled_opt)

    # Apply activation function 
    x = self.relu(x)

    # Pass output to second dense layer
    x = self.fc2(x)

    # Apply softmax activation
    x = self.softmax(x)

    return x 


# Initialize the Custom model
model = CustomHead(bert)
print('\n>> Custom Model initialized')

from transformers import AdamW

# Define the optimizer function
optimizer = AdamW(model.parameters(), lr = 1e-3)

from sklearn.utils.class_weight import compute_class_weight

# Compute the class weights, to cater the imbalance dataset
class_wts = compute_class_weight('balanced',
                                np.unique(train_labels), train_labels)

print('\nclass weights: {}'.format(class_wts))

