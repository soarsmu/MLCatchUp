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

class TestTorch(TestCase):

    def test_addbmm(self):
        # num_batches = 10

        res3 = torch.addbmm(1, res2, 0, b1, b2)
        self.assertEqual(res3, res2)

        res4 = torch.addbmm(1, res2, .5, b1, b2)
        self.assertEqual(res4, res.sum(0)[0] * 3)

        res5 = torch.addbmm(0, res2, 1, b1, b2)
        self.assertEqual(res5, res.sum(0)[0])

        res6 = torch.addbmm(.1, res2, .5, b1, b2)
        self.assertEqual(res6, res2 * .1 + res.sum(0) * .5)
