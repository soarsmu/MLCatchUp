import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from model.fc import FCNet
from torch.nn.utils.weight_norm import weight_norm

class ImgAttention(nn.Module):

    def _gated_tanh(self, x, W, W_prime):
        y_tilde = F.tanh(W(x))
        g = F.sigmoid(W_prime(x))
        y = torch.mul(y_tilde, g)
        return y, g

    def forward(self, v, q, out=False):
        logits, joint_repr = self.logits(v, q, out=out)
        logits = logits.view(-1, self.K)

        if self.att_mode == 'sigmoid':
            sig_maps = nn.functional.sigmoid(logits)
        else:
            sig_maps = nn.functional.softmax(logits)
        maps = sig_maps
        maps = maps.view(-1, self.K, 1)
        f = (v * maps).sum(1)
        return f, maps, sig_maps, logits, joint_repr
