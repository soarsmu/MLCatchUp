import torch
import torchvision
import torch.nn.functional as F
import numpy as np
from torch.nn.functional import sigmoid, tanh

class LSTMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, Wy, bias, by, old_h, old_cell):
        X = torch.concat(1, [old_h, input])
        gate_weights = torch.addmm(bias, X, weights.T)
        gates = gate_weights.chunk(4,1)


        y = torch.addmm(by, h, Wy.T)



        return prob, state, cache
