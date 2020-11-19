import torch
import numpy as np

from qpth.util import get_sizes

# TODO: Add more comments describing the math here.
# https://stanford.edu/~boyd/papers/pdf/code_gen_impl.pdf




def solve_kkt(U_Q, d, G, A, U_S, rx, rs, rz, ry, dbg=False):
    """ Solve KKT equations for the affine step"""
    nineq, nz, neq, _ = get_sizes(G, A)

    invQ_rx = torch.potrs(rx.view(-1, 1), U_Q).view(-1)
    if neq > 0:
        h = torch.cat([torch.mv(A, invQ_rx) - ry,
                       torch.mv(G, invQ_rx) + rs / d - rz], 0)
    else:
        h = torch.mv(G, invQ_rx) + rs / d - rz

    w = -torch.potrs(h.view(-1, 1), U_S).view(-1)

    g1 = -rx - torch.mv(G.t(), w[neq:])
    if neq > 0:
        g1 -= torch.mv(A.t(), w[:neq])
    g2 = -rs - w[neq:]

    dx = torch.potrs(g1.view(-1, 1), U_Q).view(-1)
    ds = g2 / d
    dz = w[neq:]
    dy = w[:neq] if neq > 0 else None

    # if rs.norm() > 0: import IPython, sys; IPython.embed(); sys.exit(-1)
    return dx, ds, dz, dy


def pre_factor_kkt(Q, G, A):
    """ Perform all one-time factorizations and cache relevant matrix products"""
    nineq, nz, neq, _ = get_sizes(G, A)

    # S = [ A Q^{-1} A^T        A Q^{-1} G^T           ]
    #     [ G Q^{-1} A^T        G Q^{-1} G^T + D^{-1} ]

    U_Q = torch.potrf(Q)
    # partial cholesky of S matrix
    U_S = torch.zeros(neq + nineq, neq + nineq).type_as(Q)

    G_invQ_GT = torch.mm(G, torch.potrs(G.t(), U_Q))
    R = G_invQ_GT
    if neq > 0:
        invQ_AT = torch.potrs(A.t(), U_Q)
        A_invQ_AT = torch.mm(A, invQ_AT)
        G_invQ_AT = torch.mm(G, invQ_AT)


    return U_Q, U_S, R


def factor_kkt(U_S, R, d):
    """ Factor the U22 block that we can only do after we know D. """
    nineq = R.size(0)
    U_S[-nineq:, -nineq:] = torch.potrf(R + torch.diag(1 / d.cpu()).type_as(d))


def factor_solve_kkt(Q, D, G, A, rx, rs, rz, ry):
    nineq, nz, neq, _ = get_sizes(G, A)


    invH_A_ = torch.potrs(A_.t(), U_H_)
    invH_g_ = torch.potrs(g_.view(-1, 1), U_H_).view(-1)

    S_ = torch.mm(A_, invH_A_)
    U_S_ = torch.potrf(S_)
    t_ = torch.mv(A_, invH_g_).view(-1, 1) - h_
    w_ = -torch.potrs(t_, U_S_).view(-1)
    v_ = torch.potrs(-g_.view(-1, 1) - torch.mv(A_.t(), w_), U_H_).view(-1)

    return v_[:nz], v_[nz:], w_[:nineq], w_[nineq:] if neq > 0 else None
