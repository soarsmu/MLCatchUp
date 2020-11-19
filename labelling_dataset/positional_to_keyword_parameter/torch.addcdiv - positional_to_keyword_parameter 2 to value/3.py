from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Sparsemax(nn.Module):

    def forward(self, input):

        sum_zs = (torch.sum(zs_sparse, dim) - 1)
        taus = Variable(torch.zeros(k_max.size()),requires_grad=False)
        taus = torch.addcdiv(taus, (torch.sum(zs_sparse, dim) - 1), k_max)
        taus = torch.unsqueeze(taus,1)
        taus_expanded = taus.expand_as(input_shift)
        output = Variable(torch.zeros(input_reshape.size()))
        output = torch.max(output, input_shift - taus_expanded)
        return output.view(-1, self.num_clusters*self.num_neurons_per_cluster), zs_sparse,taus, is_gt

