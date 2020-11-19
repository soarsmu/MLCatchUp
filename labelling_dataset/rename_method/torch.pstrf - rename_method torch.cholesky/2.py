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

SIZE = 100

class TestTorch(TestCase):

    def test_pstrf(self):
        def checkPsdCholesky(a, uplo, inplace):
            if inplace:
                u = torch.empty_like(a)
                piv = a.new(a.size(0)).int()
                kwargs = {'out': (u, piv)}
            else:
                kwargs = {}
            args = [a]

            if uplo is not None:
                args += [uplo]

            u, piv = torch.pstrf(*args, **kwargs)

            if uplo is False:
                a_reconstructed = torch.mm(u, u.t())
            else:
                a_reconstructed = torch.mm(u.t(), u)
