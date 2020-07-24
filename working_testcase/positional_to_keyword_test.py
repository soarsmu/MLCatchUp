import torch
a = torch.randn(4)
b = torch.randn(4,1)

torch.add(a, 2, b)

import torch as to

to.add(a, 2, b)

from torch import add

add(a,2,b)
add(a,b,alpha=2)