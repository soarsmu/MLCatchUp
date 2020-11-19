"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""
from cocogan_nets import *
from init import *
from helpers import get_model_list, _compute_fake_acc, _compute_true_acc
import torch
from torch import autograd
import torch.nn as nn
import os
import itertools
from visualize import make_dot
import pdb


class COCOGANGAMESTrainer(nn.Module):
  def dis_update(self, images_a, images_b, hyperparameters):
    for it, (this_true_a, this_true_b, this_fake_a, this_fake_b, in_a, in_b, fake_a, fake_b) in enumerate(itertools.izip(res_true_a, res_true_b, res_fake_a, res_fake_b, images_a, images_b, x_ba, x_ab)):
      #pdb.set_trace()
      out_true_a, out_fake_a = nn.functional.sigmoid(this_true_a), nn.functional.sigmoid(this_fake_a)
      out_true_b, out_fake_b = nn.functional.sigmoid(this_true_b), nn.functional.sigmoid(this_fake_b)

    loss.backward()
    self.dis_opt.step()
    self.dis_loss = loss.data.cpu().numpy()[0]
    return

 
  def js_regularization(self, D1_logits, D1_arg, D2_logits, D2_arg):
    D1 = nn.functional.sigmoid(D1_logits)
    D2 = nn.functional.sigmoid(D2_logits)

