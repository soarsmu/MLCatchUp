import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable
from math import sqrt 

class BatchDeterminant(autograd.Function):
    """
    batched computation of determinant
    """

    @staticmethod
    def backward(ctx, grad_output):
        LU, pivot, dets = ctx.saved_tensors

        LUinv_t = torch.zeros_like(LU)
        unit_vec = torch.zeros(LU.shape[0], LU.shape[1], dtype = LU.dtype).to(LU.device)
        
        for i in range(LU.shape[1]):
            unit_vec[:, i] = 1
            LUinv_t[:, i, :] = torch.btrisolve(unit_vec, LU, pivot)
            unit_vec[:, i] = 0            

        return grad_output.view(-1, 1, 1) * dets.view(-1, 1, 1) * LUinv_t

