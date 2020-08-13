import torch
import torch.nn as nn

class ETFPriceLoss(nn.Module):
    """
        Calculate the price part of the ETF competion score
        label type: a batch of tensors recoding ETF close price of 5 days
        output type: a batch of tensors recoding ETF close price of 5 days
    """
    def __init__(self):
        super(ETFPriceLoss, self).__init__()

    def forward(self, label, output):
        loss = torch.abs(torch.add(label, -1, output))
        loss = torch.add(label, -1, loss)
        loss = torch.div(loss, label)
        weights = torch.tensor([0.1,0.15,0.2,0.25,0.3])
        loss = torch.mul(loss, weights)
        loss = torch.sum(loss, dim=1)
        loss = torch.mul(loss, -0.5)
        loss = torch.mean(loss)
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
        pastIndices = torch.Tensor([0,1,2,3]).long()
        dataLength = torch.tensor(data[0].size()[0]).long()
        lastPrice = torch.index_select(data, 1, dataLength-1)
        pastLabel = torch.cat((lastPrice, torch.index_select(label, 1, pastIndices)), 1)
        pastOutput = torch.cat((lastPrice, torch.index_select(output, 1, pastIndices)), 1)
        varLabel = torch.sign(torch.add(label, -1, pastLabel))
        varOutput = torch.sign(torch.add(output, -1, pastOutput))
        part2 = torch.eq(varLabel, varOutput).float()
        ## total score
        loss = torch.add(part1, 1, part2)
        weights = torch.tensor([0.1,0.15,0.2,0.25,0.3])
        loss = torch.mul(loss, weights)
        loss = torch.sum(loss, dim=1)
        loss = torch.mul(loss, -0.5)
        loss = torch.mean(loss)

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
        part1 = torch.div(part1, label)
        ## variation score
        outputLength = output[0].size()[0] 
        pastLabel = data
        pastOutput = torch.cat((data[:,0:1],output[:,:-1]), 1)
        varLabel = torch.sign(torch.add(label, -1, pastLabel))
        varOutput = torch.sign(torch.add(output, -1, pastOutput))
        part2 = torch.eq(varLabel, varOutput).double()
        ## total score
        loss = torch.add(part1, 1, part2)
        loss = torch.sum(loss, dim=1)
        loss = torch.mul(loss, -0.5/outputLength)
        loss = torch.mean(loss)

        return loss