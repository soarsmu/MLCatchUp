import torch
from enum import Enum
# from block import block

from qpth.util import get_sizes, bdiag


shown_btrifact_warning = False

def pre_factor_kkt(Q, G, A):
    """ Perform all one-time factorizations and cache relevant matrix products"""
    nineq, nz, neq, nBatch = get_sizes(G, A)

    try:
        Q_LU = btrifact_hack(Q)
    except:
        raise RuntimeError("""
qpth Error: Cannot perform LU factorization on Q.
Please make sure that your Q matrix is PSD and has
a non-zero diagonal.
""")

    # S = [ A Q^{-1} A^T        A Q^{-1} G^T          ]
    #     [ G Q^{-1} A^T        G Q^{-1} G^T + D^{-1} ]
    #
    # We compute a partial LU decomposition of the S matrix
    # that can be completed once D^{-1} is known.
    # See the 'Block LU factorization' part of our website
    # for more details.

    G_invQ_GT = torch.bmm(G, G.transpose(1, 2).btrisolve(*Q_LU))
    R = G_invQ_GT.clone()
    S_LU_pivots = torch.IntTensor(range(1, 1 + neq + nineq)).unsqueeze(0) \
        .repeat(nBatch, 1).type_as(Q).int()
    if neq > 0:
        invQ_AT = A.transpose(1, 2).btrisolve(*Q_LU)
        A_invQ_AT = torch.bmm(A, invQ_AT)
        G_invQ_AT = torch.bmm(G, invQ_AT)

        LU_A_invQ_AT = btrifact_hack(A_invQ_AT)
        P_A_invQ_AT, L_A_invQ_AT, U_A_invQ_AT = torch.btriunpack(*LU_A_invQ_AT)
        P_A_invQ_AT = P_A_invQ_AT.type_as(A_invQ_AT)

        S_LU_11 = LU_A_invQ_AT[0]
        U_A_invQ_AT_inv = (P_A_invQ_AT.bmm(L_A_invQ_AT)
                           ).btrisolve(*LU_A_invQ_AT)
        S_LU_21 = G_invQ_AT.bmm(U_A_invQ_AT_inv)
        T = G_invQ_AT.transpose(1, 2).btrisolve(*LU_A_invQ_AT)
        S_LU_12 = U_A_invQ_AT.bmm(T)
        S_LU_22 = torch.zeros(nBatch, nineq, nineq).type_as(Q)
        S_LU_data = torch.cat((torch.cat((S_LU_11, S_LU_12), 2),
                               torch.cat((S_LU_21, S_LU_22), 2)),
                              1)
        S_LU_pivots[:, :neq] = LU_A_invQ_AT[1]

        R -= G_invQ_AT.bmm(T)
    else:
        S_LU_data = torch.zeros(nBatch, nineq, nineq).type_as(Q)

    S_LU = [S_LU_data, S_LU_pivots]
    return Q_LU, S_LU, R


factor_kkt_eye = None


def factor_kkt(S_LU, R, d):
    """ Factor the U22 block that we can only do after we know D. """
    nBatch, nineq = d.size()
    neq = S_LU[1].size(1) - nineq
    # TODO: There's probably a better way to add a batched diagonal.
    global factor_kkt_eye
    if factor_kkt_eye is None or factor_kkt_eye.size() != d.size():
        # print('Updating batchedEye size.')
        factor_kkt_eye = torch.eye(nineq).repeat(
            nBatch, 1, 1).type_as(R).byte()
    T = R.clone()
    T[factor_kkt_eye] += (1. / d).squeeze().view(-1)

    T_LU = btrifact_hack(T)

    global shown_btrifact_warning
    if shown_btrifact_warning or not T.is_cuda:
        # TODO: Don't use pivoting in most cases because
        # torch.btriunpack is inefficient here:
        oldPivotsPacked = S_LU[1][:, -nineq:] - neq
        oldPivots, _, _ = torch.btriunpack(
            T_LU[0], oldPivotsPacked, unpack_data=False)
        newPivotsPacked = T_LU[1]
        newPivots, _, _ = torch.btriunpack(
            T_LU[0], newPivotsPacked, unpack_data=False)

        # Re-pivot the S_LU_21 block.
        if neq > 0:
            S_LU_21 = S_LU[0][:, -nineq:, :neq]
            S_LU[0][:, -nineq:,
                    :neq] = newPivots.transpose(1, 2).bmm(oldPivots.bmm(S_LU_21))

        # Add the new S_LU_22 block pivots.
        S_LU[1][:, -nineq:] = newPivotsPacked + neq

    # Add the new S_LU_22 block.
    S_LU[0][:, -nineq:, -nineq:] = T_LU[0]
