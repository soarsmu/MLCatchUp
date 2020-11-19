"""
Reference:  Iterative Normalization: Beyond Standardization towards Efficient Whitening, CVPR 2019

- Paper:
- Code: https://github.com/huangleiBuaa/IterNorm

*****
This implementation allows the number of featur maps is not divided by the channel number of per Group. E,g. one can use group size of 64 when the channel number is 80. (64 + 16)

"""
import torch.nn
from torch.nn import Parameter

# import extension._bcnn as bcnn

__all__ = ['iterative_normalization_FlexGroup', 'IterNorm']


#
# class iterative_normalization(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, *inputs):
#         result = bcnn.iterative_normalization_forward(*inputs)
#         ctx.save_for_backward(*result[:-1])
#         return result[-1]
#
#     @staticmethod
#     def backward(ctx, *grad_outputs):
#         grad, = grad_outputs
#         grad_input = bcnn.iterative_normalization_backward(grad, ctx.saved_variables)
#         return grad_input, None, None, None, None, None, None, None


class iterative_normalization_py(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
       
        if training:
            Sigma = torch.addmm(eps, P[0], 1. / m, xc, xc.transpose(0, 1))
            # reciprocal of trace of Sigma: shape [g, 1, 1]
        
            for k in range(ctx.T):
                P[k + 1] = torch.addmm(1.5, P[k], -0.5, torch.matrix_power(P[k], 3), Sigma_N)
            saved.extend(P)
        else:
            xc = x - running_mean
            wm = running_wmat
        return Xn

    @staticmethod
    def backward(ctx, *grad_outputs):
        # g_sigma = g_sigma + g_sigma.transpose(-2, -1)
        g_x = torch.addmm(wm.mm(g_ - g_.mean(-1, keepdim=True)), g_sigma, xc)
        grad_input = g_x.view(grad.size(1), grad.size(0), *grad.size()[2:]).transpose(0, 1).contiguous()
        return grad_input, None, None, None, None, None, None, None

