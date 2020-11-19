# functions for affine transformation
import math, torch
import numpy as np
import torch.nn.functional as F

# make target * theta = source
def solve2theta(source, target):
  source, target = source.clone(), target.clone()
  oks = source[2, :] == 1
  assert torch.sum(oks).item() >= 3, 'valid points : {:} is short'.format(oks)
  if target.size(0) == 2: target = torch.cat((target, oks.unsqueeze(0).float()), dim=0)
  source, target = source[:, oks], target[:, oks]
  source, target = source.transpose(1,0), target.transpose(1,0)
  assert source.size(1) == target.size(1) == 3
  #X, residual, rank, s = np.linalg.lstsq(target.numpy(), source.numpy())
  #theta = torch.Tensor(X.T[:2, :])
  X_, qr = torch.gels(source, target)
  theta = X_[:3, :2].transpose(1, 0)
  return theta
