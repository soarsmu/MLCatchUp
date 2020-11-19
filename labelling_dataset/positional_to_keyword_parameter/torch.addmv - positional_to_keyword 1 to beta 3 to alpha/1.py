import torch

from ..function import Function, InplaceFunction

class Addmv(_BlasBase):

    def forward(self, add_vector, matrix, vector):
        return torch.addmv(self.alpha, add_vector, self.beta,
                           matrix, vector, out=output)
