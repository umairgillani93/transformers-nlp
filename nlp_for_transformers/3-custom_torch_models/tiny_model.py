import torch

class TinyModel(torch.nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()
        # linear layer one
        self.linear1 = torch.nn.Linear(100, 200)
        # activation function one
        self.act1 = torch.nn.ReLU()
        # linear layer two
        self.linear2 = torch.nn.Linear(200, 10)
        # softmax to fine the logits probabiilties
        self.softmax = torch.nn.Softmax()


    def forward(self, x):
        '''
        forward pass for custome Neural Network
        '''
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

# IMPORTANT: If a layer has M inputs and N outputs
# The weights matrix size would be M * N size matrix

# instantiate the class
tinymodel = TinyModel()
print(f'The model: {tinymodel}')
print(f'First layer: {tinymodel.linear1}')
print(f'Second layer: {tinymodel.linear2}')
print(f'Actiation func: {tinymodel.act1}')
#print(f'activation func: {tinymodel.activation}')

print([param for param in tinymodel.parameters()])

print([
    param for param in tinymodel.linear2.parameters()]
    )
