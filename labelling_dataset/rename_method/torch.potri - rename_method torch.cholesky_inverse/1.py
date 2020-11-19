import sys
import math
import random
import torch
import torch.cuda
import tempfile
import unittest
from itertools import product, chain
from functools import wraps
from common import TestCase, iter_indices, TEST_NUMPY

if TEST_NUMPY:
    import numpy as np

SIZE = 100

def skipIfNoLapack(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            fn(*args, **kwargs)
        except Exception as e:
            if 'Lapack library not found' in e.args[0]:
                raise unittest.SkipTest('Compiled without Lapack')
            raise
    return wrapper

class TestTorch(TestCase):

    @skipIfNoLapack
    def tset_potri(self):
        a=torch.Tensor(((6.80, -2.11,  5.66,  5.97,  8.23),
                        (-6.05, -3.30,  5.36, -4.44,  1.08),
                        (-0.45,  2.58, -2.70,  0.27,  9.04),
                        (8.32,  2.71,  4.35, -7.17,  2.14),
                        (-9.67, -5.14, -7.26,  6.08, -6.87))).t()

        # make sure 'a' is symmetric PSD
        a = a * a.t()

        # compute inverse directly
        inv0 = torch.inverse(a)

        # default case
        chol = torch.potrf(a)
        inv1 = torch.potri(chol)
        self.assertLessEqual(inv0.dist(inv1), 1e-12)

        # upper Triangular Test
        chol = torch.potrf(a, 'U')
        inv1 = torch.potri(chol, 'U')
        self.assertLessEqual(inv0.dist(inv1), 1e-12)

        # lower Triangular Test
        chol = torch.potrf(a, 'L')
        inv1 = torch.potri(chol, 'L')
        self.assertLessEqual(inv0.dist(inv1), 1e-12)
