import os 
import numpy as np 
import pandas as pd 
import textwrap 
import matplotlib.pyplot as plt 
from pprint import pprint
from transformers import pipeline 

print('\nimports successful')

path = os.getenv('HOME') + '/fillmask_pretrained_weights/'
print('\nPaht: {}'.format(path))

df = pd.read_csv('bbc_text_cls.csv')

print(df.head())


labels = set(df['labels'])

df['labels'].hist()
#plt.show()

print(df['labels'].value_counts())

print('\nlabels: {}'.format(labels))

texts = df[df['labels'] == 'business']['text']
print('\ntexts: {}'.format(texts.head()))


# Replicates the same results each time
np.random.seed(1234)

# Randomly choose any text document
c = np.random.choice(texts.shape[0])
print('\nC is: {}'.format(c))

# Returns the row on cth location
doc = texts.iloc[c]

# Print out the chosen article
print(textwrap.fill(doc, replace_whitespace=False, fix_sentence_endings=True))  

# instantiate model object
print('\n >> Instantiating model')
mlm = pipeline('fill-mask', path)

mlm('Bombardier chief to leave <mask>')
text = 'shares in <mask> and plane-making'

mlm(text)
pprint(mlm(text))
