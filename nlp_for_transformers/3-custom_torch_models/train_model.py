import os 
import torch
import torch.nn as nn
import torchvision
#from torch.utils.tensorboard import SymmaryWriter
import torchvision.transforms as transforms
import torch.nn.functional as F

# Pytorch Tensorboard support
#from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

transform = transforms.Compose(
        [transforms.ToTensor(), # convert data observations to tensors
        transforms.Normalize((0.5,), (0.5,))]) # Normalize the data observations

# Downlaod datasets
print(f' >> Downloading datasets..')
training_set = torchvision.datasets.FashionMNIST(os.getenv('HOME') + '/datasets/', train=True
                                                        ,transform=transform, download=True)
validation_set = torchvision.datasets.FashionMNIST(os.getenv('HOME') + '/datasets/', train=False,
                                                    transform=transform, download=True)


# Prepare datasets for Training and Validation with Dataloaders
train_loader = torch.utils.data.DataLoader(
        training_set, batch_size=4,
        shuffle=True,
        num_workers=2
        )

valid_loader = torch.utils.data.DataLoader(
        validation_set,
        shuffle=True,
        num_workers=2
        )

print('\nLoaders done')

# class labesl
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# Show the split sizes
print(f'Training set has {len(training_set)} sizes')
print(f'Validation set has {len(validation_set)} sizes')

# building custom model
class GarmentClassifier(nn.Module):
    def __init__(self):
        super(GarmentClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.Pool(F.relu(self.conv1(x)))
        x = self.Pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

model = GarmentClassifier()

# Set loss function
loss_fn = torch.nn.CrossEntropyLoss()
dummy_outputs = torch.rand(4,10)
dummy_labels = torch.tensor([1,5,3,7])

loss_fn = loss_fn(dummy_outputs, dummy_labels)

print(f'Total loss for the batch is: {loss_fn.item()}')

