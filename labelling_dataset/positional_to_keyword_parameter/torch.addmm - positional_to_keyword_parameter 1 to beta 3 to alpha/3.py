import torch
from torch import sparse

import itertools
import random
import unittest
from common import TestCase, run_tests
from numbers import Number

SparseTensor = sparse.DoubleTensor


class TestSparse(TestCase):

    def test_mm(self):
        def test_shape(di, dj, dk):
            x, _, _ = self._gen_sparse(2, 20, [di, dj])
            t = torch.randn(di, dk)
            y = torch.randn(dj, dk)
            alpha = random.random()
            beta = random.random()

            expected = torch.addmm(alpha, t, beta, x.to_dense(), y)
            res = torch.addmm(alpha, t, beta, x, y)
            self.assertEqual(res, expected)

            expected = torch.addmm(t, x.to_dense(), y)
            res = torch.addmm(t, x, y)
        test_shape(10, 100, 100)
        test_shape(100, 1000, 200)
        test_shape(64, 10000, 300)

    def test_saddmm(self):
        def test_shape(di, dj, dk):
            x = self._gen_sparse(2, 20, [di, dj])[0]
            t = self._gen_sparse(2, 20, [di, dk])[0]
            y = torch.randn(dj, dk)
            alpha = random.random()
            beta = random.random()

            expected = torch.addmm(alpha, t.to_dense(), beta, x.to_dense(), y)
            