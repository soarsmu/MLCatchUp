import sys
import os
import math
import random
import torch
import torch.cuda
import tempfile
import unittest
import warnings
from itertools import product, combinations
from common import TestCase, iter_indices, TEST_NUMPY, run_tests, download_file, skipIfNoLapack

if TEST_NUMPY:
    import numpy as np

SIZE = 100


class TestTorch(TestCase):
    @staticmethod
    def _test_btrisolve(self, cast):
        a = torch.FloatTensor((((1.3722, -0.9020),
                                (1.8849, 1.9169)),
                               ((0.7187, -1.1695),
                                (-0.0139, 1.3572)),
                               ((-1.6181, 0.7148),
                                (1.3728, 0.1319))))
        b = torch.FloatTensor(((4.02, 6.19),
                               (-1.56, 4.00),
                               (9.81, -4.09)))
        a, b = cast(a), cast(b)
        info = cast(torch.IntTensor())
        LU_data, pivots = a.btrifact(info=info)
        self.assertEqual(info.abs().sum(), 0)
        x = torch.btrisolve(b, LU_data, pivots)
        b_ = torch.bmm(a, x.unsqueeze(2)).squeeze()
        self.assertEqual(b_, b)
