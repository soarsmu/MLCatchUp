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

  
def factor_kkt(S_LU, R, d):

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
