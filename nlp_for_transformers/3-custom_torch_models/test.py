import os
import torch

print(f'torch version: {torch.__version__}')

for epoch in range(100):
    
    running_loss = 0.0

    for j, data in enumerate(training_dataloader):
        inputs, labels = data
        outputs = 


