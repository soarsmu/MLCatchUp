"""
Functions for principal component analysis (PCA)

---------------------------------------------------------------------

pca
    principal component analysis (singular value decomposition)

---------------------------------------------------------------------

Copyright 2018 Facebook Inc.
Copyright 2019 Evgenii Zheltonozhskii
All rights reserved.

"Software" means the fbpca software distributed by Facebook Inc.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions, and the following disclaimer in
  the documentation and/or other materials provided with the
  distribution.

* Neither the name Facebook nor the names of its contributors may be
  used to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

Additional grant of patent rights:

Facebook hereby grants you a perpetual, worldwide, royalty-free,
non-exclusive, irrevocable (subject to the termination provision
below) license under any rights in any patent claims owned by
Facebook, to make, have made, use, sell, offer to sell, import, and
otherwise transfer the Software. For avoidance of doubt, no license
is granted under Facebook's rights in any patent claims that are
infringed by (i) modifications to the Software made by you or a third
party, or (ii) the Software in combination with any software or other
technology provided by you or a third party.

The license granted hereunder will terminate, automatically and
without notice, for anyone that makes any claim (including by filing
any lawsuit, assertion, or other action) alleging (a) direct,
indirect, or contributory infringement or inducement to infringe any
patent: (i) by Facebook or any of its subsidiaries or affiliates,
whether or not such claim is related to the Software, (ii) by any
party if such claim arises in whole or in part from any software,
product or service of Facebook or any of its subsidiaries or
affiliates, whether or not such claim is related to the Software, or
(iii) by any party relating to the Software; or (b) that any right in
any patent claim of Facebook is invalid or unenforceable.
"""

import numpy as np
import torch


def my_lu(A, permute_l=False):
    if len(A.shape) == 2:
        A = A.unsqueeze(0)
    M, N = A.shape[1], A.shape[2]
    K = int(np.min((M, N)))
    if A.shape[1] != A.shape[2]:
        m = int(np.max((A.shape[1], A.shape[2])))
        ten = torch.eye(m).unsqueeze(0).repeat(A.shape[0], 1, 1).to(A)
        ten[:, : A.shape[1], :A.shape[2]] = A
        A = ten
    A_LU, pivots = torch.btrifact(A)
    P, A_L, A_U = torch.btriunpack(A_LU, pivots)
    P, A_L, A_U = P[:, :M, :M].squeeze(), A_L[:, :M, :K].squeeze(), A_U[:, :K, :N].squeeze()
    if permute_l:
        return torch.matmul(P, A_L), A_U
    else:
        return P, A_L, A_U


def pca(A, k=6, raw=False, n_iter=2, l=None):
    """
    Principal component analysis.

    Constructs a nearly optimal rank-k approximation U diag(s) Va to A,
    centering the columns of A first when raw is False, using n_iter
    normalized power iterations, with block size l, started with a
    min(m,n) x l random matrix, when A is m x n; the reference PCA_
    below explains "nearly optimal." k must be a positive integer <=
    the smaller dimension of A, n_iter must be a nonnegative integer,
    and l must be a positive integer >= k.

    The rank-k approximation U diag(s) Va comes in the form of a
    singular value decomposition (SVD) -- the columns of U are
    orthonormal, as are the rows of Va, and the entries of s are all
    nonnegative and nonincreasing. U is m x k, Va is k x n, and
    len(s)=k, when A is m x n.

    Increasing n_iter or l improves the accuracy of the approximation
    U diag(s) Va; the reference PCA_ below describes how the accuracy
    depends on n_iter and l. Please note that even n_iter=1 guarantees
    superb accuracy, whether or not there is any gap in the singular
    values of the matrix A being approximated, at least when measuring
    accuracy as the spectral norm || A - U diag(s) Va || of the matrix
    A - U diag(s) Va (relative to the spectral norm ||A|| of A, and
    accounting for centering when raw is False).

    Notes
    -----
    To obtain repeatable results, reset the seed for the pseudorandom
    number generator.

    The user may ascertain the accuracy of the approximation
    U diag(s) Va to A by invoking diffsnorm(A, U, s, Va), when raw is
    True. The user may ascertain the accuracy of the approximation
    U diag(s) Va to C(A), where C(A) refers to A after centering its
    columns, by invoking diffsnormc(A, U, s, Va), when raw is False.

    Parameters
    ----------
    A : array_like, shape (m, n)
        matrix being approximated
    k : int, optional
        rank of the approximation being constructed;
        k must be a positive integer <= the smaller dimension of A,
        and defaults to 6
    raw : bool, optional
        centers A when raw is False but does not when raw is True;
        raw must be a Boolean and defaults to False
    n_iter : int, optional
        number of normalized power iterations to conduct;
        n_iter must be a nonnegative integer, and defaults to 2
    l : int, optional
        block size of the normalized power iterations;
        l must be a positive integer >= k, and defaults to k+2

    Returns
    -------
    U : ndarray, shape (m, k)
        m x k matrix in the rank-k approximation U diag(s) Va to A or
        C(A), where A is m x n, and C(A) refers to A after centering
        its columns; the columns of U are orthonormal
    s : ndarray, shape (k,)
        vector of length k in the rank-k approximation U diag(s) Va to
        A or C(A), where A is m x n, and C(A) refers to A after
        centering its columns; the entries of s are all nonnegative and
        nonincreasing
    Va : ndarray, shape (k, n)
        k x n matrix in the rank-k approximation U diag(s) Va to A or
        C(A), where A is m x n, and C(A) refers to A after centering
        its columns; the rows of Va are orthonormal

    Examples
    --------
    # >>> from fbpca import diffsnorm, pca
    # >>> from numpy.random import uniform
    # >>> from scipy.linalg import svd
    # >>>
    # >>> A = uniform(low=-1.0, high=1.0, size=(100, 2))
    # >>> A = A.dot(uniform(low=-1.0, high=1.0, size=(2, 100)))
    # >>> (U, s, Va) = svd(A, full_matrices=False)
    # >>> A = A / s[0]
    # >>>
    # >>> (U, s, Va) = pca(A, 2, True)
    # >>> err = diffsnorm(A, U, s, Va)
    # >>> print(err)

    This example produces a rank-2 approximation U diag(s) Va to A such
    that the columns of U are orthonormal, as are the rows of Va, and
    the entries of s are all nonnegative and are nonincreasing.
    diffsnorm(A, U, s, Va) outputs an estimate of the spectral norm of
    A - U diag(s) Va, which should be close to the machine precision.

    References
    ----------
    .. [PCA] Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp,
             Finding structure with randomness: probabilistic
             algorithms for constructing approximate matrix
             decompositions, arXiv:0909.4061 [math.NA; math.PR], 2009
             (available at `arXiv <http://arxiv.org/abs/0909.4061>`_).

    See also
    --------
    diffsnorm, diffsnormc, eigens, eigenn
    """

    if l is None:
        l = k + 2

    (m, n) = A.shape

    assert k > 0
    assert k <= min(m, n)
    assert n_iter >= 0
    assert l >= k

    # Promote the types of integer data to float data.
    dtype = (A * 1.0).dtype
    device = A.device

    if raw:

        #
        # SVD A directly if l >= m/1.25 or l >= n/1.25.
        #
        if l >= m / 1.25 or l >= n / 1.25:
            (U, s, Va) = torch.svd(A, compute_uv=True)
            #
            # Retain only the leftmost k columns of U, the uppermost
            # k rows of Va, and the first k entries of s.
            #
            return U[:, :k], s[:k], Va.transpose(0, 1)[:k, :]

        if m >= n:

            #
            # Apply A to a random matrix, obtaining Q.
            #
            Q = torch.empty((n, l)).uniform_(-1.0, 1.0).to(dtype=dtype, device=device)
            Q = torch.matmul(A, Q)

            #
            # Form a matrix Q whose columns constitute a
            # well-conditioned basis for the columns of the earlier Q.
            #
            if n_iter == 0:
                Q, _ = torch.qr(Q)
            if n_iter > 0:
                Q, _ = my_lu(Q, permute_l=True)

            #
            # Conduct normalized power iterations.
            #
            for it in range(n_iter):
                old_q_shape = Q.shape
                Q = torch.matmul(Q.transpose(0, 1), A).transpose(0, 1)

                Q, _ = my_lu(Q, permute_l=True)

                Q = torch.matmul(A, Q)

                if it + 1 < n_iter:
                    Q, _ = my_lu(Q, permute_l=True)
                else:
                    Q, _ = torch.qr(Q)

                assert old_q_shape == Q.shape

            #
            # SVD Q'*A to obtain approximations to the singular values
            # and right singular vectors of A; adjust the left singular
            # vectors of Q'*A to approximate the left singular vectors
            # of A.
            #
            QA = torch.matmul(Q.transpose(0, 1), A)
            (R, s, Va) = torch.svd(QA, compute_uv=True)
            U = torch.matmul(Q, R)

            #
            # Retain only the leftmost k columns of U, the uppermost
            # k rows of Va, and the first k entries of s.
            #
            return U[:, :k], s[:k], Va.transpose(0, 1)[:k, :]

        if m < n:

            #
            # Apply A' to a random matrix, obtaining Q.
            #
            R = torch.empty((l, m)).uniform_(-1.0, 1.0).to(dtype=dtype, device=device)

            Q = torch.matmul(R, A).transpose(0, 1)

            #
            # Form a matrix Q whose columns constitute a
            # well-conditioned basis for the columns of the earlier Q.
            #
            if n_iter == 0:
                Q, _ = torch.qr(Q)
            if n_iter > 0:
                Q, _ = my_lu(Q, permute_l=True)

            #
            # Conduct normalized power iterations.
            #
            for it in range(n_iter):

                Q = torch.matmul(A, Q)
                Q, _ = my_lu(Q, permute_l=True)

                Q = torch.matmul(Q.transpose(0, 1), A).transpose(0, 1)

                if it + 1 < n_iter:
                    Q, _ = my_lu(Q, permute_l=True)
                else:
                    Q, _ = torch.qr(Q)

            #
            # SVD A*Q to obtain approximations to the singular values
            # and left singular vectors of A; adjust the right singular
            # vectors of A*Q to approximate the right singular vectors
            # of A.
            #
            (U, s, Ra) = torch.svd(torch.matmul(A, Q))
            Va = torch.matmul(Ra, Q.transpose(0, 1))

            #
            # Retain only the leftmost k columns of U, the uppermost
            # k rows of Va, and the first k entries of s.
            #
            return U[:, :k], s[:k], Va[:k, :]

    if not raw:

        #
        # Calculate the average of the entries in every column.
        #
        c = A.sum(dim=0) / m
        c = c.view((1, n))

        #
        # SVD the centered A directly if l >= m/1.25 or l >= n/1.25.
        #
        if l >= m / 1.25 or l >= n / 1.25:
            (U, s, Va) = torch.svd(A - torch.matmul(torch.ones((m, 1), dtype=dtype, device=device), c), compute_uv=True)
            #
            # Retain only the leftmost k columns of U, the uppermost
            # k rows of Va, and the first k entries of s.
            #
            return U[:, :k], s[:k], Va.transpose(0, 1)[:k, :]

        if m >= n:

            #
            # Apply the centered A to a random matrix, obtaining Q.
            #
            R = torch.empty((n, l)).uniform_(-1.0, 1.0).to(dtype=dtype, device=device)

            Q = torch.matmul(A, R) - torch.matmul(torch.ones((m, 1), dtype=dtype, device=device), torch.matmul(c, R))

            #
            # Form a matrix Q whose columns constitute a
            # well-conditioned basis for the columns of the earlier Q.
            #
            if n_iter == 0:
                Q, _ = torch.qr(Q)
            if n_iter > 0:
                Q, _ = my_lu(Q, permute_l=True)

            #
            # Conduct normalized power iterations.
            #
            for it in range(n_iter):

                Q = torch.matmul(Q.transpose(0, 1), A) \
                    - torch.matmul(torch.matmul(Q.transpose(0, 1), torch.ones((m, 1), dtype=dtype, device=device)), c)
                Q = Q.transpose(0, 1)
                Q, _ = my_lu(Q, permute_l=True)

                Q = torch.matmul(A, Q) - torch.matmul(torch.ones((m, 1), dtype=dtype, device=device),
                                                      torch.matmul(c, Q))

                if it + 1 < n_iter:
                    Q, _ = my_lu(Q, permute_l=True)
                else:
                    Q, _ = torch.qr(Q)

            #
            # SVD Q' applied to the centered A to obtain
            # approximations to the singular values and right singular
            # vectors of the centered A; adjust the left singular
            # vectors to approximate the left singular vectors of the
            # centered A.
            #
            QA = torch.matmul(Q.transpose(0, 1), A) \
                 - torch.matmul((torch.matmul(Q.transpose(0, 1), torch.ones((m, 1), dtype=dtype, device=device))), c)
            (R, s, Va) = torch.svd(QA)
            U = torch.matmul(Q, R)

            #
            # Retain only the leftmost k columns of U, the uppermost
            # k rows of Va, and the first k entries of s.
            #
            return U[:, :k], s[:k], Va.transpose(0, 1)[:k, :]

        if m < n:

            #
            # Apply the adjoint of the centered A to a random matrix,
            # obtaining Q.
            #
            R = torch.empty((l, m)).uniform_(-1.0, 1.0).to(dtype=dtype, device=device)

            Q = torch.matmul(R, A) - torch.matmul(torch.matmul(R, torch.ones((m, 1), dtype=dtype, device=device)), c)
            Q = Q.transpose(0, 1)

            #
            # Form a matrix Q whose columns constitute a
            # well-conditioned basis for the columns of the earlier Q.
            #
            if n_iter == 0:
                Q, _ = torch.qr(Q)
            if n_iter > 0:
                Q, _ = my_lu(Q, permute_l=True)

            #
            # Conduct normalized power iterations.
            #
            for it in range(n_iter):

                Q = torch.matmul(A, Q) - torch.matmul(torch.ones((m, 1), dtype=dtype, device=device),
                                                      torch.matmul(c, Q))
                Q, _ = my_lu(Q, permute_l=True)

                Q = torch.matmul(Q.transpose(0, 1), A) \
                    - torch.matmul(torch.matmul(Q.transpose(0, 1), torch.ones((m, 1), dtype=dtype, device=device)), c)
                Q = Q.transpose(0, 1)

                if it + 1 < n_iter:
                    Q, _ = my_lu(Q, permute_l=True)
                else:
                    Q, _ = torch.qr(Q)

            #
            # SVD the centered A applied to Q to obtain approximations
            # to the singular values and left singular vectors of the
            # centered A; adjust the right singular vectors to
            # approximate the right singular vectors of the centered A.
            #
            (U, s, Ra) = torch.svd(torch.matmul(A, Q) - torch.matmul(torch.ones((m, 1), dtype=dtype, device=device),
                                                                     torch.matmul(c, Q)),
                                   compute_uv=True)
            Va = torch.matmul(Ra, Q.transpose(0, 1))

            #
            # Retain only the leftmost k columns of U, the uppermost
            # k rows of Va, and the first k entries of s.
            #
            return U[:, :k], s[:k], Va[:k, :]
