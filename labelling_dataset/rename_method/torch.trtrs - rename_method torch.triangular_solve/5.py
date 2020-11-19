"""Utility functions of linear algebra."""

import torch


def cho_solve(cho_C, b):
    """Compute tensor $C^{-1} b$ from cholesky factor.

    ----
    Parameters:
        cho_C: (N x N) lower triangular tensor where cho_C cho_C^T = C
        b: (N x L) tensor
    ----
    Outputs:
        C^{-1} b
    ----
    Note:
        Gradient of potrs is not supperted yet in pytorch 0.4.1
        # return torch.potrs(b, cho_C, upper=False)
    """
    tmp, _ = torch.trtrs(b, cho_C, upper=False)
    tmp2, _ = torch.trtrs(tmp, cho_C.t(), upper=True)
    return tmp2


def cho_solve_AXB(a, cho_C, b):
    """Compute tensor $a C^{-1} b$ from cholesky factor.

    ----
    Parameters:
        a: (M x N) tensor
        cho_C: (N x N) lower triangular tensor where cho_C cho_C^T = C
        b: (N x L) tensor
    ----
    Outputs:
        a C^{-1} b
    """
    left, _ = torch.trtrs(a.t(), cho_C, upper=False)
    right, _ = torch.trtrs(b, cho_C, upper=False)

    return torch.mm(left.t(), right)
