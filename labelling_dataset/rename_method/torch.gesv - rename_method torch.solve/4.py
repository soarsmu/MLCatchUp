"""
THIS FILE IS ADAPTED FROM FINN'S GPS
"""

import torch
import numpy as np


def gauss_fit_joint_prior(pts, mu0, Phi, m, n0, dwts, d1, d2, sig_reg):

    sigma = (N * empsig + Phi + (N * m) / (N + m) * torch.ger(mun - mu0, mun - mu0)) / (N + n0)
    sigma = 0.5 * (sigma + sigma.T)
    # Add sigma regularization.
    sigma += sig_reg
    # Conditioning to get dynamics.
    fd = torch.gesv(sigma[:d1, :d1],
                    sigma[:d1, d1:d1 + d2]).t()
    fc = mu[d1:d1 + d2] - fd.matmul(mu[:d1])
    dynsig = sigma[d1:d1 + d2, d1:d1 + d2] - \
             fd.matmul(sigma[:d1, :d1]).matmul(fd.t())
    dynsig = 0.5 * (dynsig + dynsig.t())
    return fd, fc, dynsig
