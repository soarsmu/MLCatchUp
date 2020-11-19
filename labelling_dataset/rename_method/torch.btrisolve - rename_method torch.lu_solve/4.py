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
import types
import re
from torch._utils_internal import get_file_path, get_file_path_2
from torch.utils.dlpack import from_dlpack, to_dlpack
from torch._utils import _rebuild_tensor
from torch._six import inf, nan, string_classes
from itertools import product, combinations
from functools import reduce
from torch import multiprocessing as mp
from common_utils import TestCase, iter_indices, TEST_NUMPY, TEST_SCIPY, TEST_MKL, \
    TEST_LIBROSA, run_tests, download_file, skipIfNoLapack, suppress_warnings, \
    IS_WINDOWS, PY3, NO_MULTIPROCESSING_SPAWN, skipIfRocm
from multiprocessing.reduction import ForkingPickler

if TEST_NUMPY:
    import numpy as np

if TEST_SCIPY:
    from scipy import signal

if TEST_LIBROSA:
    import librosa

class FilelikeMock(object):
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
        LU_data, pivots, info = a.btrifact_with_info()
        self.assertEqual(info.abs().sum(), 0)
        x = torch.btrisolve(b, LU_data, pivots)
        b_ = torch.bmm(a, x.unsqueeze(2)).squeeze()
        self.assertEqual(b_, b)
