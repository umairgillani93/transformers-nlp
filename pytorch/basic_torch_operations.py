import torch 
print('successful')

# tow dimentional array
x = torch.empty(2,3)
# four dimentional array
y = torch.empty(
    size=(3,3,3,3)
    )

print('X: {}'.format(x))
print('X shape: {}'.format(x))
print('Y: {}'.format(y))
print('Y shape: {}'.format(y.shape))

# size of tensor 
print('X size: {}'.format(x.size))

# check data type
print('X data type: {}'.format(x.dtype))

# specify data type of tensor 
zero_tensor = torch.zeros(2,2, dtype=torch.float16)

# passed data
data = [
    [1,2,3],
    [4,5,6]
    ]

# constructing from data
x = torch.tensor(data)
print('X shape: {}'.format(x.shape))
print('X type: {}'.format(type(x)))
print('X size: {}'.format(x.size()))


x = [1,2,3]
y = [4,5,6]
x_tensor = torch.tensor(x, dtype=torch.float16)
y_tensor = torch.tensor(y, dtype=torch.float16)

# element-wise addition
z = x + y
z_ = torch.add(x, y)
print(z)
print(z_)

# element-wise substraction
z = x - y
z_ = torch.sub(x,y)
print(z)
print(z_)

# element-wise multiplication
z = x * y 
print(z)


ones_tensor = torch.ones(
    (2,2), requires_grad=True)

print(a)

ones_mean = ones_tensor + 5

c = ones_mean.mean()

print(ones_mean, c)

c.backward()

print(ones_tensor.grad)
