 #Tensors - similar to numpy ndarrays also can be used on a GPU accelerated computing
# -*- coding: utf-8 -*-

from __future__ import print_function
import torch

print (x+y)
print(torch.add(x, y))

# providing an output tensor as argument

torch.add(x, y, out=result)
print(result)


print(a)
#tensor([ 0.6939,  0.4375, -1.5540, -1.0776])
torch.add(a, 20)
#tensor([20.6939, 20.4375, 18.4460, 18.9224])

torch.add(a, 10, b)
#tensor([[  2.7695,   3.3930,   4.3672,   4.1450],
#        [-18.6971, -18.0736, -17.0994, -17.3216],
#        [ -6.7845,  -6.1610,  -5.1868,  -5.4090],
#        [ -8.9902,  -8.3667,  -7.3925,  -7.6147]])

a = torch.randn(4)
torch.ceil(a)
#tensor([-0., -1., -1.,  1.])





