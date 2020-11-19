import torch
import torch.nn as nn

class ETFPriceLoss(nn.Module):

    def forward(self, label, output):
        loss = torch.abs(torch.add(label, -1, output))
        loss = torch.add(label, -1, loss)
        loss = torch.div(loss, label)
        weights = torch.tensor([0.1,0.15,0.2,0.25,0.3])
        return loss

class ETFLoss(nn.Module):
    """
        Calculate the ETF competion score
        label type: a batch of tensors recoding ETF close price of 5 days
        output type: a batch of tensors recoding ETF close price of 5 days
        data type: a batch of tensors recoding ETF close price, the last element
        in each tensor should be the last close price 
    """
    def __init__(self):
        super(ETFLoss, self).__init__()

    def forward(self, label, output, data):
        ## price score
        part1 = torch.abs(torch.add(label, -1, output))
        part1 = torch.add(label, -1, part1)
        part1 = torch.div(part1, label)
        ## variation score
        ## total score
        loss = torch.add(part1, 1, part2)
        weights = torch.tensor([0.1,0.15,0.2,0.25,0.3])
        return loss

class ETFLSTMLoss(nn.Module):
    """
        Calculate the average ETF competion score for n days, default n=19
        label type: a batch of tensors recoding ETF close price of n days
        output type: a batch of tensors recoding ETF close price of n days
        data type: a batch of tensors recoding ETF close price, the first element
        in each tensor should be the first close price used for prediction
    """
    def __init__(self):
        super(ETFLSTMLoss, self).__init__()

    def forward(self, label, output, data):
        ## price score
        part1 = torch.abs(torch.add(label, -1, output))
        part1 = torch.add(label, -1, part1)
       
        ## total score
        loss = torch.add(part1, 1, part2)
        return loss