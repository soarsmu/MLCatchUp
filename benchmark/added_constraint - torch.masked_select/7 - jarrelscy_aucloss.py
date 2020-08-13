import torch.nn.functional as F
import numpy as np
import torch

# Presumes cuda

def torchDiffWMW( target,pred, gamma=0.5, p = 2):    
    mask = target.type(torch.cuda.ByteTensor)
    x = torch.masked_select(pred, 1+(-1)*mask).view(1,-1)
    y = torch.masked_select(pred, mask).view(-1,1)
    xn = x.expand(y.size(0), x.size(1))
    yn = y.expand(y.size(0), x.size(1))
    mask = (xn-yn) < gamma
    ur = torch.pow((xn-yn-gamma),p)*mask.type(torch.cuda.FloatTensor)
    return torch.sum(ur) / torch.sum(target) / torch.sum(1-target)

def torchSigWMW( target,pred, beta=2):    
    mask = target.type(torch.cuda.ByteTensor)
    x = torch.masked_select(pred, 1+(-1)*mask).view(1,-1)
    y = torch.masked_select(pred, mask).view(-1,1)
    xn = x.expand(y.size(0), x.size(1))
    yn = y.expand(y.size(0), x.size(1))
    ur = F.sigmoid(-beta*(xn-yn))
    return torch.sum(ur) / torch.sum(target) / torch.sum(1-target)
    
def torchPairSigWMW( target,pred, pred_class, other_class,beta=2):    
    mask = target.type(torch.cuda.ByteTensor)
    oc = (target == other_class)
    pc = (target == pred_class)
    x = torch.masked_select(pred, oc).view(1,-1)
    y = torch.masked_select(pred, pc).view(-1,1)
    xn = x.expand(y.size(0), x.size(1))
    yn = y.expand(y.size(0), x.size(1))
    ur = F.sigmoid(-beta*(xn-yn))
    return torch.sum(ur) / torch.sum(oc).float() / torch.sum(pc).float()
    
def npSigWMW( target,pred, beta=2):    
    x = pred[target == 0.0].reshape((1,-1))
    y = pred[target == 1.0].reshape((-1,1))    
    xn = np.repeat(x, y.shape[0], 0)
    yn = np.repeat(y, x.shape[1], 1)
    ur = 1.0 / (1.0 + np.exp(beta*(xn-yn)))    
    return ur.sum() / float(np.sum(target)) / float(np.sum(1.0 - target))

def npDiffWMW(target,pred, gamma=0.5, p = 2):     
    x = pred[target == 0.0].reshape((1,-1))
    y = pred[target == 1.0].reshape((-1,1))    
    xn = np.repeat(x, y.shape[0], 0)
    yn = np.repeat(y, x.shape[1], 1)
    mask = (xn-yn) < gamma    
    ur = np.power((xn-yn-gamma),p)*mask
    return ur.sum() / float(np.sum(target)) / float(np.sum(1.0 - target))
    
def npAUC(target,pred, gamma=0.0, p = 2):     
    x = pred[target == 0.0].reshape((1,-1))
    y = pred[target == 1.0].reshape((-1,1))    
    xn = np.repeat(x, y.shape[0], 0)
    yn = np.repeat(y, x.shape[1], 1)
    mask = (xn-yn) < gamma    
    ur = mask
    return ur.sum() / float(np.sum(target)) / float(np.sum(1.0 - target))
    
    
def ovoWMWLoss(pred, target, beta, num_classes=10):
    ret = 0
    for i in range(num_classes):
        for k in range(num_classes):
            if i!=k:
                ret += torchPairSigWMW(target, pred[:,i], i, k, beta)
    return -ret / (float(num_classes) * (float(num_classes) - 1.0) )

def WMWLoss(pred, target, beta, num_classes=10):
    ret = 0
    for i in range(num_classes):
        ret += torchSigWMW((target == i).long().float(), pred[:,i], beta)
    return -ret / float(num_classes)
    
    
