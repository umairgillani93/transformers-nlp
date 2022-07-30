import torch 

a= torch.ones(
    (2,2), requires_grad = True)

print(a.grad)
b = a + 5 
c = b.mean()
print('Propagating backward')
c.backward()

print('A gradient: {}'.format(a.grad))
