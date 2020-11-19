#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import ceil
import cv2
import Networks


class Weighted_least_squares(nn.Module):
    def __init__(self, size, nclasses, order, no_cuda, reg_ls=0, use_cholesky=False):

        if not self.use_cholesky:
            # Z_inv = [torch.inverse(matrix) for matrix in torch.unbind(Z)]
            # Z_inv = torch.stack(Z_inv)
            Z_inv = torch.inverse(Z)
            X = torch.bmm(Y0.transpose(1, 2), torch.mul(W0, x_map))
            beta0 = torch.bmm(Z_inv, X)
        else:
            # cholesky
            # TODO check this
            beta0 = []
            X = torch.bmm(Y0.transpose(1, 2), torch.mul(W0, x_map))
            for image, rhs in zip(torch.unbind(Z), torch.unbind(X)):
                R = torch.potrf(image)
                opl = torch.trtrs(rhs, R.transpose(0, 1))
            beta0 = torch.cat((beta0), 1).transpose(0, 1).unsqueeze(2)
