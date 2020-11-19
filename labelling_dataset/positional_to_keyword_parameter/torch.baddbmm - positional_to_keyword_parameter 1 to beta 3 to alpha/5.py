"""
Reference:  Iterative Normalization: Beyond Standardization towards Efficient Whitening, CVPR 2019

- Paper:
- Code: https://github.com/huangleiBuaa/IterNorm
"""
import torch.nn
import torch.nn.functional as F
from torch.nn import Parameter

# import extension._bcnn as bcnn

__all__ = ['iterative_normalization', 'IterNorm']

class iterative_normalization_py(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):

        saved = []
        if training:
            # calculate centered activation by subtracted mini-batch mean

            P[0] = torch.eye(d).to(X).expand(ctx.g, d, d)
            Sigma = torch.baddbmm(eps, P[0], 1. / m, xc, xc.transpose(1, 2))
            # reciprocal of trace of Sigma: shape [g, 1, 1]
            saved.append(Sigma_N)
            for k in range(ctx.T):
                P[k + 1] = torch.baddbmm(1.5, P[k], -0.5, torch.matrix_power(P[k], 3), Sigma_N)
            saved.extend(P)
            wm = P[ctx.T].mul_(rTr.sqrt())  # whiten matrix: the matrix inverse of Sigma, i.e., Sigma^{-1/2}
            running_mean.copy_(momentum * mean + (1. - momentum) * running_mean)
            running_wmat.copy_(momentum * wm + (1. - momentum) * running_wmat)
        else:
            xc = x - running_mean
            wm = running_wmat
        xn = wm.matmul(xc)
        Xn = xn.view(X.size(1), X.size(0), *X.size()[2:]).transpose(0, 1).contiguous()
        ctx.save_for_backward(*saved)
        return Xn

    @staticmethod
    def backward(ctx, *grad_outputs):
        g_sn += g_P
        # g_sn = g_sn * rTr.sqrt()
        g_tr = ((-sn.matmul(g_sn) + g_wm.transpose(-2, -1).matmul(wm)) * P[0]).sum((1, 2), keepdim=True) * P[0]
        g_sigma = (g_sn + g_sn.transpose(-2, -1) + 2. * g_tr) * (-0.5 / m * rTr)
        # g_sigma = g_sigma + g_sigma.transpose(-2, -1)
        g_x = torch.baddbmm(wm.matmul(g_ - g_.mean(-1, keepdim=True)), g_sigma, xc)
        grad_input = g_x.view(grad.size(1), grad.size(0), *grad.size()[2:]).transpose(0, 1).contiguous()
        return grad_input, None, None, None, None, None, None, None
