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

class FilelikeMock(object):

    def test_addbmm(self):

        res3 = torch.addbmm(1, res2, 0, b1, b2)
        self.assertEqual(res3, res2)

        res4 = torch.addbmm(1, res2, .5, b1, b2)
        self.assertEqual(res4, res.sum(0, False) * 3)

        res5 = torch.addbmm(0, res2, 1, b1, b2)
        self.assertEqual(res5, res.sum(0, False))

        res6 = torch.addbmm(.1, res2, .5, b1, b2)
        self.assertEqual(res6, res2 * .1 + (res.sum(0) * .5))
