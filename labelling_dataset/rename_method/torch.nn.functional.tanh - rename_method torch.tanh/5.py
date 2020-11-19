
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class WhatTheNet(torch.nn.Module):

    def forward(self, x):
        # x = self.bn1(x)
        x = torch.nn.functional.tanh(self.fc1(x))
        x = torch.nn.functional.tanh(self.fc2(x))
        x2 = torch.nn.functional.relu(self.fc3(x))
        x = x + x2
        x = torch.nn.functional.tanh(self.fc21(x))
        x = torch.nn.functional.tanh(self.fc4(x))
        x = torch.nn.functional.tanh(self.fc41(x))
        x = self.fc5(x)
        
        
        return x
    

class WhatTheNet2(torch.nn.Module):
     

    def forward(self, x):
        # x = self.bn1(x)
        x = torch.nn.functional.tanh(self.fc1(x))
        x = torch.nn.functional.tanh(self.fc2(x))
        x = torch.nn.functional.tanh(self.fc3(x))
        x = torch.nn.functional.tanh(self.fc4(x))
        x = self.fc5(x)
        
        
        return x
     