import torch

from ..function import Function, InplaceFunction
from .utils import maybe_unexpand, check_onnx_broadcast

class Addmv(InplaceFunction):

    @staticmethod
    def forward(ctx, add_vector, matrix, vector, alpha=1, beta=1, inplace=False):
        ctx.alpha = alpha
        ctx.beta = beta
        ctx.add_vector_size = add_vector.size()
        ctx.save_for_backward(matrix, vector)
        output = _get_output(ctx, add_vector, inplace=inplace)
        return torch.addmv(alpha, add_vector, beta,
                           matrix, vector, out=output)