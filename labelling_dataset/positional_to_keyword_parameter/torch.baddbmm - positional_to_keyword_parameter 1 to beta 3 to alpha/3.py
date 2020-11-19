import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# waiting to solve
class IndexLinear(nn.Linear):
    def forward(self, embedding, indices=None):
        for idx, e, i in zip(range(len(embedding)),embedding,indices):
        # for idx in range(len(embedding)):
        #     e = embedding[idx]
        #     i = indices[idx]
            weight = torch.index_select(self.weight, 0, i.view(-1)).view(1, i.size(0), -1).transpose(1, 2)
            bias = torch.index_select(self.bias, 0, i.view(-1)).view(1, 1, i.size(0))
            try:
                out = torch.baddbmm(1, bias, 1, e.view(1,1,l), weight).view(li).data.numpy()# bias + embedding * weight
                outs[idx] = out
            except:
                print("outs")
        return Variable(torch.from_numpy(outs)).float().view(len(embedding),1,li)

    def reset_parameters(self):
        init_range = 0.1
        self.bias.data.fill_(0)
        self.weight.data.uniform_(-init_range, init_range)
