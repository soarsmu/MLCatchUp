import torch
from settings import float_type
from copy import deepcopy
import itertools


def batch_det(A, keepdim=True):
    # Input A is B x N x N tensor, this returns determinants for each NxN matrix based on batch LU factorisation

    # Testing:
    # B = 2000
    # N = 20
    # A = torch.randn(B, N, N)
    # A = (A + A.transpose(-1,-2))/2 # Symmetric
    # torch.allclose(apply_along_axis(torch.det, A, dim=0), batch_det(A))

    # Based on what https://pytorch.org/docs/stable/_modules/torch/functional.html - btriunpack does:
    # The diagonals of each A_LU are the diagonals of U (so det(U) = prod(diag(A_LU[n])))
    # The diagonals of L are not given, but are fixed to all 1s (so det(L) = 1)
    # The pivots determine the permutation matrix that left-multiplies A (i.e. switches rows)
    # Therefore to get the final determinant, we need to get det(L)*det(U) * (-1^[how many times P switches rows])

    sz = A.shape

    if sz[-1] != sz[-2]:
        raise('Error: can only take determinant of batches of square matrices')

    # Check if we get a batch of 1x1 matrices, then just return them
    if sz[-2] == 1:
        return A

    # Check if we get a batch of 2x2 matrices, then just compute the determinant by hand
    if sz[-2] == 2:
        return batch_det_2d(A, keepdim)

    # if multibatchaxes:
    A = A.view(-1, A.size(-2), A.size(-1))

    A_LU, pivots = torch.btrifact(A)

    # detL = 1
    detU = batch_diag(A_LU).prod(1)
    detP = (-1. * A.new_ones(A.size(0))).pow(
        ((pivots - (torch.arange(A.size(1), dtype=torch.int, device=A.device) + 1).expand(A.size(0), A.size(1))) != 0).sum(1).float())

    if keepdim:
        return (detU * detP).view(*sz[:-2], 1, 1)
    else:
        return (detU * detP).view(*sz[:-2])