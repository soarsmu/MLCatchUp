import torch
from torch import sparse

import itertools
import functools
import random
import unittest
from common import TestCase, run_tests
from common_cuda import TEST_CUDA
from test_torch import TestTorch
from numbers import Number

class TestSparse(TestCase):

    @cpu_only
    def test_mm(self):
        def test_shape(di, dj, dk):
            res = torch.addmm(alpha, t, beta, x, y)
            expected = torch.addmm(alpha, t, beta, self.safeToDense(x), y)
            self.assertEqual(res, expected)

            res = torch.addmm(t, x, y)
            expected = torch.addmm(t, self.safeToDense(x), y)
            self.assertEqual(res, expected)

            res = torch.mm(x, y)
            expected = torch.mm(self.safeToDense(x), y)
            self.assertEqual(res, expected)

        test_shape(10, 100, 100)
        test_shape(100, 1000, 200)
        test_shape(64, 10000, 300)

    @cpu_only
    def test_saddmm(self):
        def test_shape(di, dj, dk):
            expected = torch.addmm(alpha, self.safeToDense(t), beta, self.safeToDense(x), y)
            self.assertEqual(self.safeToDense(res), expected)

            expected = torch.addmm(self.safeToDense(t), self.safeToDense(x), y)
            self.assertEqual(self.safeToDense(res), expected)
