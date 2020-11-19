from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import sys
sys.path.append("../")
from util.matutil import *
from util.batchutil import *

"""
Modified by
Shiwei Lan @ CalTech, 2018
version 1.3
"""

class VAE(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(VAE, self).__init__()

#         self.iK = Variable(torch.inverse(self.K.data))
        self.iK = torch.potri(self.Kh)

    def encode(self, x):
        """ p(z|x)
            C = D + uu'
        """
        h1 = self.relu(self.fc0(x))
        enc_mean = self.fc21(h1)
        enc_covh = self.fc22(h1)
        #enc_cov_2 = self.fc23(h1)
        return enc_mean, enc_covh
