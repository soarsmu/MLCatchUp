class B:
    def other_funcaa(self, a,b,c):
        return b + c

class A:
    def other_funcaa(self, a, b, c):
        return a + b



object_a = A()
c = object_a.other_funcaa(3, 4, 0)

def other_func(a):
    a.other_funcaa(1,2,3)

import torch

a = torch.tensor([4.0, 3.0])
b = torch.tensor([2.0, 2.0])

torch.floor_divide(input=a, other=b)

import torch as to

to.floor_divide(input=a, other=b)

from torch import floor_divide
floor_divide(input=a, other=b)