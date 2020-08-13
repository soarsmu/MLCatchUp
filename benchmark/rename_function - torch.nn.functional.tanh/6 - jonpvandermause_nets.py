import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


# -----------------------------------------------------------------------------
#                           define networks
# -----------------------------------------------------------------------------

class OneLayer(nn.Module):
    def __init__(self, nodes1):
        super(OneLayer, self).__init__()

        self.lin1 = nn.Linear(4, nodes1)
        self.lin2 = nn.Linear(nodes1, 1)

    def forward(self, x):
        # preprocessing step to make the function periodic
        x = torch.tensor([torch.sin(x[0][0]), torch.cos(x[0][0]),
                          torch.sin(x[0][1]), torch.cos(x[0][1])])
        x = F.tanh(self.lin1(x))
        x = self.lin2(x)
        return x


class TwoLayer(nn.Module):
    def __init__(self, nodes1, nodes2):
        super(TwoLayer, self).__init__()

        self.lin1 = nn.Linear(4, nodes1)
        self.lin2 = nn.Linear(nodes1, nodes2)
        self.lin3 = nn.Linear(nodes2, 1)

    def forward(self, x):
        # preprocessing step to make the function periodic
        x = torch.tensor([torch.sin(x[0][0]), torch.cos(x[0][0]),
                          torch.sin(x[0][1]), torch.cos(x[0][1])])
        x = F.tanh(self.lin1(x))
        x = F.tanh(self.lin2(x))
        x = self.lin3(x)
        return x


class ThreeLayer(nn.Module):
    def __init__(self, nodes1, nodes2, nodes3):
        super(ThreeLayer, self).__init__()

        self.lin1 = nn.Linear(4, nodes1)
        self.lin2 = nn.Linear(nodes1, nodes2)
        self.lin3 = nn.Linear(nodes2, nodes3)
        self.lin4 = nn.Linear(nodes3, 1)

    def forward(self, x):
        # preprocessing step to make the function periodic
        x = torch.tensor([torch.sin(x[0][0]), torch.cos(x[0][0]),
                          torch.sin(x[0][1]), torch.cos(x[0][1])])
        x = F.tanh(self.lin1(x))
        x = F.tanh(self.lin2(x))
        x = F.tanh(self.lin3(x))
        x = self.lin4(x)
        return x
