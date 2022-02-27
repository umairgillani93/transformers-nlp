import torch 
from torch.optim import AdamW
import transformers
from transformers import Trainer
from transformers import TrainingArguments
from transformers import AutoTokenizer
#from transformers import DataCollectorWithPadding
from transformers import AutoModel, AutoModelForSequenceClassification
from datasets import load_dataset

print('import successful')

def tokenize_func(exm):
  return tokenizer(
      exm['sentence1'], exm['sentence2'],
      truncation = True
      )

# define checkpoint
checkpoint = "bert-base-uncased"

#define tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# define model; checkpoint for model and tokennizer MUST be same
training_argument = TrainingArguments('test-trainer')

# tokenizer.convert_ids_to_tokens(ids) -> converts the ids back to original tokens
raw_dataset = load_dataset('glue', 'mrpc')

raw_train_dataset = raw_dataset['train'][0]
#print([k for k in raw_train_dataset.keys()])

tokenized_dataset = raw_dataset.map(tokenize_func, batched = True)

# define model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# define trainer
trainer = Trainer(
    model,
    training_argument,
    train_dataset = tokenized_dataset['train'],
    eval_dataset = tokenized_dataset['test'],
    tokenizer = tokenizer
    )

trainer.train()
