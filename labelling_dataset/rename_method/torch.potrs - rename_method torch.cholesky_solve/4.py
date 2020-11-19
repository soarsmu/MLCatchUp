import sys
import io
import os
import math
import random
import operator
import copy
import shutil
import torch
import torch.cuda
import tempfile
import unittest
import warnings
import pickle
import gzip
from torch._utils_internal import get_file_path, get_file_path_2
from torch.utils.dlpack import from_dlpack, to_dlpack
from torch._utils import _rebuild_tensor
from itertools import product, combinations
from functools import reduce
from torch import multiprocessing as mp
from common import TestCase, iter_indices, TEST_NUMPY, TEST_SCIPY, TEST_MKL, \
    run_tests, download_file, skipIfNoLapack, suppress_warnings, IS_WINDOWS, PY3

if TEST_NUMPY:
    import numpy as np

if TEST_SCIPY:
    from scipy import signal

class FilelikeMock(object):
 
    @skipIfNoLapack
    def test_potrs(self):
        a = torch.Tensor(((6.80, -2.11, 5.66, 5.97, 8.23),
                          (-6.05, -3.30, 5.36, -4.44, 1.08),
                          (-0.45, 2.58, -2.70, 0.27, 9.04),
                          (8.32, 2.71, 4.35, -7.17, 2.14),
                          (-9.67, -5.14, -7.26, 6.08, -6.87))).t()
        b = torch.Tensor(((4.02, 6.19, -8.22, -7.57, -3.03),
                          (-1.56, 4.00, -8.67, 1.75, 2.86),
                          (9.81, -4.09, -4.57, -8.61, 8.99))).t()

        # make sure 'a' is symmetric PSD
        a = torch.mm(a, a.t())

        # upper Triangular Test
        U = torch.potrf(a)
        x = torch.potrs(b, U)
        self.assertLessEqual(b.dist(torch.mm(a, x)), 1e-12)

        # lower Triangular Test
        L = torch.potrf(a, False)
        x = torch.potrs(b, L, False)
        self.assertLessEqual(b.dist(torch.mm(a, x)), 1e-12)
