# -*- coding: utf-8 -*-

import torch

def pairwise_sub(a,b):
    column = a.unsqueeze(2)
    row = b.unsqueeze(1)
    return torch.sub(column,row)
