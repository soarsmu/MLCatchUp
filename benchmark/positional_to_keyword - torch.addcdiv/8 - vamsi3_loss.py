import torch
from .module import Module

class BCEWithLogitsLoss(Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        input = input.reshape(-1, 1)
        target = target.reshape(-1, 1)
        input_max, _ = torch.max(input, dim=1, keepdim=True)
        a = (-input_max).exp()
        z = a + (-input - input_max).exp()
        loss = input_max + z.log() + input * (1 - target)
        if self.reduction == 'mean':
            loss = loss.mean(0)
        self.save_cache(a, z)
        return loss

    def backward(self, input, target):
        input = input.reshape(-1, 1)
        target = target.reshape(-1, 1)
        a, z = self.load_cache()
        return torch.addcdiv(-target, 1, a, z)