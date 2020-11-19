# perform full, exact GP regression
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from tqdm import trange
import numpy as np
import pickle
import matplotlib.pyplot as plt

"""this implementation follows Algorithm 2.1 in Rasmussen and Williams"""

class GP(nn.Module):

    def get_LL(self, train_inputs, train_outputs):
        # form the kernel matrix Knn 
        Knn = self.get_K(train_inputs, train_inputs)

        # cholesky decompose
        L = torch.potrf(Knn + torch.exp(self.logsigman2)*torch.eye(train_inputs.shape[0]) + self.jitter*torch.eye(Knn.size()[0]), upper=False) # lower triangular decomposition
        Lslashy = torch.trtrs(train_outputs, L, upper=False)[0]
        alpha = torch.trtrs(Lslashy, torch.transpose(L,0,1))[0]

        # get log marginal likelihood
        LL = -0.5*torch.dot(train_outputs, torch.squeeze(alpha)) - torch.sum(torch.log(torch.diag(L))) - (train_inputs.shape[0]/2)*torch.log(torch.Tensor([2*3.1415926536]))
        return LL

    def joint_posterior_predictive(self, train_inputs, train_outputs, test_inputs, noise=False):
        # return the joint posterior, which is a multivariate Gaussian
        no_test = test_inputs.shape[0]
        # get training inputs covariance matrix
        Knn = self.get_K(train_inputs, train_inputs)
        
        # cholesky decompose
        L = torch.potrf(Knn + torch.exp(self.logsigman2)*torch.eye(train_inputs.shape[0]) + self.jitter*torch.eye(Knn.size()[0]), upper=False) # lower triangular decomposition
        Lslashy = torch.trtrs(train_outputs, L, upper=False)[0]
        


def gaussian_KL(mu0, mu1, Sigma0, Sigma1):
    # calculate the KL divergence between multivariate Gaussians KL(0||1)
    no_dims = Sigma0.shape[0]

    L1 = torch.potrf(Sigma1 + self.jitter*torch.eye(Sigma.size()[0]), upper=False)
    L1slashSigma = torch.trtrs(Sigma0 ,L1 ,upper=False)[0]
    SigmainvSigma = torch.trtrs(L1slashSigma ,L1.transpose(0,1))[0]
    trace_term = torch.trace(SigmainvSigma)

    mu_diff = mu1 - mu0
    v = torch.trtrs(mu_diff, L1, upper=False)[0]
    quadratic_term = v.transpose(0,1) @ v

    L0 = torch.potrf(Sigma0 + self.jitter*torch.eye(Sigma0.size()[0]), upper=False)
    logdet_term = 2*torch.sum(torch.log(torch.diag(L1))) - 2*torch.sum(torch.log(torch.diag(L0)))

    KL = 0.5*(trace_term + quadratic_term - no_dims + logdet_term)
    return KL
