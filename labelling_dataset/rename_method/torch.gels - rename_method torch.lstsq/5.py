import argparse
import os
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
cudnn.benchmark = True
torch.cuda.set_device(0)

def create_dic(A, M=50, N=10, Lmin=1, Lstep=1, Lmax=49, Epsilon=0.1, mode=0):
   

    # Adjust the length of D to meet the error requirement
    while ((global_error_D > (Epsilon * Epsilon)) and (l < Lmax + 1)):


        X_a_tmp, _ = torch.gels(A, D)
        X_a = X_a_tmp[0:l,:]

        # Add a filter
        threshold = 0.0
        for i in range(X_a.shape[0]):
            for j in range(X_a.shape[1]):
                if abs(X_a[i][j]) < threshold:
                    X_a[i][j] = 0.0

        for i in range(0, N):

            tmp = torch.mm(D, X_a)[:,i] - A[:,i]
            error_D = error_D + torch.sum(tmp * tmp)
