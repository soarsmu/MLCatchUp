import torch
import torch.nn as nn
import torch.nn.functional as F
from libs.args.args import argchoice, argignore
import numpy as np
import pdb
import utils

from modules import ReversibleFlow, Network, AffineFlowStep, Unitary

# from hessian import jacobian

class GlowPrediction(nn.Module):
    rhs = Y.matmul(O) #b.sum(1) 
    z = torch.trtrs(rhs.t(), U, transpose=False, upper=False)[0]
    x0 = torch.trtrs(z, U, transpose=True, upper=False)[0]
  
    xtemp = x0
    yhat = [C.matmul(xtemp)]
    for i in range(x.size(1) - 1):
      xtemp = A.matmul(xtemp)
 