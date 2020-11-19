"""
Orthogonalization by Newton’s Iteration
"""
import torch.nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable
from typing import List
from torch.autograd.function import once_differentiable

class ONINorm(torch.nn.Module):
   
    def forward(self, weight: torch.Tensor):
        for t in range(self.T):
            #B[t + 1] = torch.baddbmm(1.5, B[t], -0.5, torch.matrix_power(B[t], 3), S)
            B[t + 1] = torch.baddbmm(1.5, B[t], -0.5, self.matrix_power3(B[t]), S)
        W = B[self.T].matmul(Zc).div_(norm_S.sqrt())
        #print(W.matmul(W.transpose(1,2)))
        # W = oni_py.apply(weight, self.T, ctx.groups)
        return W.view_as(weight)

    def extra_repr(self):
        fmt_str = ['T={}'.format(self.T)]
        if self.norm_groups > 1:
            fmt_str.append('groups={}'.format(self.norm_groups))
        return ', '.join(fmt_str)


class ONINorm_colum(torch.nn.Module):

    def forward(self, weight: torch.Tensor):
        assert weight.shape[0] % self.norm_groups == 0
        for t in range(self.T):
            #B[t + 1] = torch.baddbmm(1.5, B[t], -0.5, torch.matrix_power(B[t], 3), S)
            B[t + 1] = torch.baddbmm(1.5, B[t], -0.5, self.matrix_power3(B[t]), S)
        W = Zc.matmul(B[self.T]).div_(norm_S.sqrt())
        #print(W.matmul(W.transpose(1,2)))
        # W = oni_py.apply(weight, self.T, ctx.groups)
        return W.view_as(weight)


