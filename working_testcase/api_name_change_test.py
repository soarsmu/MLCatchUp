import torch

a = torch.randn(4)
torch.tanh(a)


import torch as to
temp = to
temp.tanh(a)

torch.tanh(a).__str__()

b = torch.tanh(a)
b.__str__()