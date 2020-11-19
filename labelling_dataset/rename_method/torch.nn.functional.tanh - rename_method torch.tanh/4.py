import torch

class GatedTanh(torch.nn.Module):

    def forward(self, *inputs):
        forward_out = torch.nn.functional.tanh(self.linear_forward(*inputs))
        gate_out = torch.nn.functional.sigmoid(self.gate(*inputs))
        return forward_out * gate_out