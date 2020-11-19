import torch

from ..function import Function, InplaceFunction
from .utils import maybe_unexpand


class Addbmm(InplaceFunction):

    @staticmethod
    def forward(ctx, add_matrix, batch1, batch2, alpha=1, beta=1, inplace=False):
        ctx.alpha = alpha
        ctx.beta = beta
        ctx.add_matrix_size = add_matrix.size()
        ctx.save_for_backward(batch1, batch2)
        output = _get_output(ctx, add_matrix, inplace=inplace)
        return torch.addbmm(alpha, add_matrix, beta,
                            batch1, batch2, out=output)