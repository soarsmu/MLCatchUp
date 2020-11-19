#!/usr/bin/env python3

import torch


def woodbury_factor(low_rank_mat, shift):
    r"""
    Given a low rank (k x n) matrix V and a shift, returns the
    matrix R so that

    .. math::

        \begin{equation*}
            R = (I_k + 1/shift VV')^{-1}V
        \end{equation*}

    to be used in solves with (V'V + shift I) via the Woodbury formula
    """
    k = low_rank_mat.size(-2)
    shifted_mat = low_rank_mat.matmul(low_rank_mat.transpose(-1, -2) / shift.unsqueeze(-1))

    shifted_mat = shifted_mat + torch.eye(k, dtype=shifted_mat.dtype, device=shifted_mat.device)

    R = torch.potrs(low_rank_mat, torch.cholesky(shifted_mat, upper=True))
    return R
