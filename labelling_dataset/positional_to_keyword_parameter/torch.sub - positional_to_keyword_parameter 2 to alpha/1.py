# shared element for rcnn
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.functions import *


class FCSubtract(nn.Module):
    def __init__(self, D_in, D_out):
        super(FCSubtract, self).__init__()
        self.dense = nn.Linear(D_in, D_out)

    def forward(self, input_1, input_2):
        res_sub = torch.sub(input_1, input_2)
        res_sub_mul = torch.mul(res_sub, res_sub)
        out = self.dense(res_sub_mul)
        return F.relu(out)
