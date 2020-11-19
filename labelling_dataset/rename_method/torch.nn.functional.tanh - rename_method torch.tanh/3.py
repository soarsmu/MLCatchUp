import torch
import numpy as np
import torch.nn.functional as functional
from torch.autograd import Variable
from mnist_read import read_mnist
import matplotlib.pyplot as plt


class Net(torch.nn.Module):

    def forward(self, x):
        x = self.norm_layer(x)
        x = functional.tanh(self.hidden_layer(x))
        y_prediction = functional.log_softmax(self.output_layer(x))
        return y_prediction

