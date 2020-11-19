import torch
import math
import numpy as np
from matplotlib import cm

def calIntsAcc(gt_i, pred_i, data_batch=1):
    n, c, h, w = gt_i.shape
    pred_i  = pred_i.view(n, c, h, w)
    ref_int = gt_i[:, :3].repeat(1, gt_i.shape[1] // 3, 1, 1)
    gt_i  = gt_i / ref_int
    scale = torch.gels(gt_i.view(-1, 1), pred_i.view(-1, 1))
    ints_ratio = (gt_i - scale[0][0] * pred_i).abs() / (gt_i + 1e-8)
    ints_error = torch.stack(ints_ratio.split(3, 1), 1).mean(2)
    return {'ints_ratio': ints_ratio.mean().item()}, ints_error.squeeze()
    
