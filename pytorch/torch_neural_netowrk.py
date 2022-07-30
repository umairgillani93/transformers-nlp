import torch 

print('pytorch version: {}'.format(torch.__version__))

# input tensor
X = torch.Tensor([
    [1,0,1,0],
    [0,0,1,0],
    [1,0,0,0]])

# labels
y = torch.Tensor([
  [1],[0],[1]])
  

print('INput tensor: {}'.format(X))
print('Output labels: {}'.format(y))

# activaton function
def sigmoid(x):
  return 1 / (1+torch.exp(-x))

# sigmoid gradient
def sigmoid_gradient(x):
  return sigmoid(x) * (1 * sigmoid(x))


epochs = 500
lr = 0.1
input_layer_units = X.shape[1] # number of rows of X
hidden_layer_units = 3
output_units = 1

# weights and biase initiliazation
wh = torch.randn(input_layer_units, hidden_layer_units).type(torch.FloatTensor)
bh = torch.randn(1,hidden_layer_units).type(torch.FloatTensor)
wout=torch.randn(hidden_layer_units, output_units)
bout=torch.randn(1, output_units)

for i in range(epochs):
  # forward pass
  hidden_layer_input1 = torch.mm(X,wh)
  hidden_layer_input = hidden_layer_input1 + bh
  hidden_layer_activations = sigmoid(hidden_layer_input)

  output_layer_input1 = torch.mm(hidden_layer_activations, wout)
  output_layer_input = output_layer_input1 + bout
  output = sigmoid(output_layer_input)

  # Backpropagation
  E = y - output
  slope_output_layer = sigmoid_gradient(output)
  slope_hidden_layer = sigmoid_gradient(hidden_layer_activations)
  d_output = E * slope_output_layer
  error_at_hidden_layer = torch.mm(d_output, wout.t())
  d_hidden_layer = error_at_hidden_layer * slope_hidden_layer
  wout += torch.mm(hidden_layer_activations.t(), d_output) * lr
  bout += d_output.sum() * lr
  wh += torch.mm(X.t(), d_hidden_layer) * lr
  bh += d_output.sum() * lr

print('actual :\n', y, '\n')
print('predicted :\n', output)
  
