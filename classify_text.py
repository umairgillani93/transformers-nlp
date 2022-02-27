import torch 
from torch.optim import AdamW
import transformers
from transformers import AutoTokenizer
from transformers import DataCollectorWithPadding
from transformers import AutoModel, AutoModelForSequenceClassification
from datasets import load_dataset

print('import successful')

def tokenize_func(exm):
  return tokenizer(exm['sentence1'], exm['sentence2'],
      truncation = True)

# define checkpoint
checkpoint = "bert-base-uncased"

#define tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# define model; checkpoint for model and tokennizer MUST be same
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# define input data
#sequences = [
#    "I've been waiting for you my whole life.",
#    "My mom is amazing."
#    ]

# creating the batch
#batch = tokenizer(sequences, padding = True, truncation = True,
#                  return_tensors = 'pt')

# tokenizer.convert_ids_to_tokens(ids) -> converts the ids back to original tokens
raw_dataset = load_dataset('glue', 'mrpc')

raw_train_dataset = raw_dataset['train'][0]
#print([k for k in raw_train_dataset.keys()])

tokenized_dataset = raw_dataset.map(tokenize_func, batched = True)

data_collector = DataCollectorWithPadding(tokenizer=tokenizer)
samples = tokenized_dataset['train'][:8]
print('samples: {}'.format(samples))
samples_input_ids = {k:v for k,v in samples.items() if k not in ['idx', 'sentence1', 'sentence2']}
print('samples input ids: {}'.format(sample_input_ids))


#batch['labels'] = torch.tensor([1, 1])
#optimizer = AdamW(model.parameters())
#loss = model(**batch).loss
#loss.backward()
#optimizer.step()
