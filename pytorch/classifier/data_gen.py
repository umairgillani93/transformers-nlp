import os 
import pandas as pd 

data = {
    'text': ['clinical impression found',
            'clinical impression not found',
            'clinical impression found',
            'clinical impression not found',
            'clinical impression found',
            'clinical impression not found',
            'clinical impression found',
            'clinical impression not found',
            'clinical impresion found',
            'clinical impression not found'
           ], 

    'labels': [1, 0, 1, 0,1, 0, 1,0, 1, 0]

    }

assert len(data['text']) == len(data['labels'])

df = pd.DataFrame(data)

print(f'Dataframe: {df}')

df.to_csv(r'./train.csv')
