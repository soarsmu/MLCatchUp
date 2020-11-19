import sys
import os
import math
import random
import copy
import torch
import torch.cuda
import tempfile
import unittest
import warnings
from itertools import product, combinations
from common import TestCase, iter_indices, TEST_NUMPY, run_tests, download_file, skipIfNoLapack, \
    suppress_warnings

if TEST_NUMPY:
    import numpy as np

SIZE = 100


class TestTorch(TestCase):

    def test_pstrf(self):
        def checkPsdCholesky(a, uplo, inplace):
            if inplace:
                u = torch.Tensor(a.size())
                piv = torch.IntTensor(a.size(0))
                kwargs = {'out': (u, piv)}
            else:
                kwargs = {}
            args = [a]

            if uplo is not None:
                args += [uplo]

            u, piv = torch.pstrf(*args, **kwargs)
