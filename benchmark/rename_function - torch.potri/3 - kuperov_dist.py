import math
from typing import List, Tuple

import torch
from torch.distributions import Transform, constraints
from torch.distributions.utils import lazy_property


class ReciprocalTransform(Transform):
    """Reciprocal transform y = 1/x."""

    sign = 1

    def _call(self, x):
        return 1 / x  # P(x=0) = 0 for diffuse distributions

    def _inverse(self, y):
        return 1. / y  # P(y=0) = 0 for diffuse distributions

    def log_abs_det_jacobian(self, x, y):
        return 2 * torch.log(y)


def InvGamma(a, b):
    """Returns an inverse gamma distribution.

    Args:
        a: concentration parameter
        b: scale (inverse of rate) parameter
    """
    return torch.distributions.TransformedDistribution(
        torch.distributions.Gamma(concentration=a, rate=b), ReciprocalTransform()
    )


def _batch_mv(bmat, bvec):
    r"""
    Performs a batched matrix-vector product, with compatible but different batch shapes.

    This function takes as input `bmat`, containing :math:`n \times n` matrices, and
    `bvec`, containing length :math:`n` vectors.

    Both `bmat` and `bvec` may have any number of leading dimensions, which correspond
    to a batch shape. They are not necessarily assumed to have the same batch shape,
    just ones which can be broadcasted.
    """
    n = bvec.size(-1)
    batch_shape = _get_batch_shape(bmat, bvec)

    # to conform with `torch.bmm` interface, both bmat and bvec should have `.dim() == 3`
    bmat = bmat.expand(batch_shape + (n, n)).reshape((-1, n, n))
    bvec = bvec.unsqueeze(-1).expand(batch_shape + (n, 1)).reshape((-1, n, 1))
    return torch.bmm(bmat, bvec).view(batch_shape + (n,))


def _get_batch_shape(bmat, bvec):
    r"""
    Given a batch of matrices and a batch of vectors, compute the combined `batch_shape`.
    """
    try:
        vec_shape = torch._C._infer_size(bvec.shape, bmat.shape[:-1])
    except RuntimeError:
        raise ValueError(
            "Incompatible batch shapes: vector {}, matrix {}".format(
                bvec.shape, bmat.shape
            )
        )
    return torch.Size(vec_shape[:-1])


def _sparse_diag(A):
    """Compute diag for a sparse matrix"""
    idxs = A._indices()
    return A._values()[idxs[0] == idxs[1]]


class MVNPrecisionTril(torch.distributions.Distribution):
    """
    Multivariate normal distribution parameterized by a mean vector and lower-
    triangular cholesky factor of the precision matrix.

    Args:
        loc (Tensor):            mean of the distribution
        precision_tril (Tensor): lower-triangular factor of precision, with
                                 positive-valued diagonal

    For simplicity does not support batches for now.
    """

    support = constraints.real
    arg_constraints = {"precision_tril": constraints.lower_cholesky}
    has_rsample = False

    def __init__(self, loc, precision_tril, validate_args=None):
        self.loc, self.precision_tril = loc, precision_tril
        event_shape = torch.Size(loc.shape[-1:])
        if precision_tril.dim() < 2:
            raise ValueError(
                "precision_tril matrix must be at least two-dimensional, "
                "with optional leading batch dimensions"
            )
        vec_shape = torch._C._infer_size(loc.shape, precision_tril.shape[:-1])
        batch_shape = torch.Size(vec_shape[:-1])
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def mean(self):
        return self.loc

    @lazy_property
    def variance(self):
        return torch.potri(self.loc, upper=False)  # is this diffable?

    @lazy_property
    def scale_tril(self):
        if self.precision_tril.is_sparse:
            return torch.potri(self.precision_tril.to_dense())
        else:
            return torch.potri(self.precision_tril, upper=False)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        diff = (value - self.loc).unsqueeze(1)
        if self.precision_tril.is_sparse:
            # note Mchol is dense
            Mchol = torch.matmul(self.precision_tril.t(), diff)
            log_det = -sum(abs(torch.log(_sparse_diag(self.precision_tril))))
        else:
            Mchol = torch.matmul(self.precision_tril.t(), diff)
            log_det = -torch.diag(self.precision_tril).abs().log().sum(-1)
        M = torch.matmul(Mchol.t(), Mchol)
        return -0.5 * (M + self.loc.size(-1) * math.log(2 * math.pi)) - log_det

    def entropy(self):
        if self.precision_tril.is_sparse:
            log_det = -_sparse_diag(self.precision_tril).abs().log().sum(-1)
        else:
            log_det = -torch.diag(self.precision_tril).abs().log().sum(-1)
        H = 0.5 * (1.0 + math.log(2 * math.pi)) * self._event_shape[0] + log_det
        return H

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = self.loc.new(*shape).normal_()
        return self.loc + _batch_mv(self.scale_tril, eps)


class Improper(torch.distributions.Distribution):

    support = constraints.real
    has_rsample = False

    def log_prob(self, value):
        return 0.


class InvWishart(torch.distributions.Distribution):
    """Inverse-Wishart distribution, which is a distribution over real-valued
    positive-definite matrices.

    """

    arg_constraints = {
        "df": constraints.positive,
        "scale": constraints.positive_definite,
    }
    support = constraints.real
    has_rsample = False

    def __init__(self, df, scale):
        """Create inverse Wishart distribution.

        Args:
            df:    degrees of freedom parameter
            scale: positive-definite scale matrix
        """
        self.df, self.scale, self.p = df, scale, scale.shape[0]

    @property
    def mean(self):
        return self.scale / (self.df - self.p - 1)

    # @property
    # def variance(self):
    #     pass
    # TODO: implement variance

    def log_prob(self, value):
        _p = self.p
        assert value.shape == (_p, _p), f"value should be {_p}x{_p} psd"
        X_inv = value.inverse()
        _df = self.df
        return (
            +0.5 * _df * torch.slogdet(self.scale)[1]
            - 0.5 * _df * _p * math.log(2)
            - special.multigammaln(0.5 * _df, _p)
            - 0.5 * (_df + _p + 1) * torch.slogdet(value)[1]  # |X|^{-(nu+p+1)/2}
            - 0.5 * torch.trace(self.scale @ X_inv)
        )


# def logpdf_cholesky(L, df, Psi):
#     """Inverse-wishart log density with Cholesky parameterization.
#
#     Computes p(X | Psi, df) where X ~ W^-1(Psi, df), where X = L@L.T and L is
#     lower-triangular
#
#     Returns:
#       log density of L
#     """
#     p = Psi.shape[0]
#     L_inv = np.linalg.inv(np.tril(L))  # better to backsolve
#     return (
#             + 0.5 * df * np.linalg.slogdet(Psi)[1]
#             - 0.5 * df * p * np.log(2)
#             - special.multigammaln(0.5 * df, p)
#             - (df + p + 1) * np.sum(np.log(np.diag(L)))  # |X|^{-(nu+p+1)/2}
#             - 0.5 * np.trace(np.dot(np.dot(Psi, L_inv.T), L_inv))
#     )
