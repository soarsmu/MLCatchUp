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
from common import TestCase, iter_indices, TEST_NUMPY, TEST_SCIPY, TEST_MKL, \
    TEST_LIBROSA, run_tests, download_file, skipIfNoLapack, suppress_warnings, \
    IS_WINDOWS, PY3, NO_MULTIPROCESSING_SPAWN, skipIfRocm
from multiprocessing.reduction import ForkingPickler

if TEST_NUMPY:
    import numpy as np

if TEST_SCIPY:
    from scipy import signal

if TEST_LIBROSA:
    import librosa

SIZE = 100

can_retrieve_source = True
with warnings.catch_warnings(record=True) as warns:
    with tempfile.NamedTemporaryFile() as checkpoint:
        x = torch.save(torch.nn.Module(), checkpoint)
        for warn in warns:
            if "Couldn't retrieve source code" in warn.message.args[0]:
                can_retrieve_source = False
                break


class FilelikeMock(object):
  
    def test_addmv(self):
        types = {
            'torch.DoubleTensor': 1e-8,
            'torch.FloatTensor': 1e-4,
        }
        for tname, _prec in types.items():
            t = torch.randn(10).type(tname)
            m = torch.randn(10, 100).type(tname)
            v = torch.randn(100).type(tname)
            res1 = torch.addmv(t, m, v)
            res2 = torch.zeros(10).type(tname)
            

        # Test 0-strided
        for tname, _prec in types.items():
            t = torch.randn(1).type(tname).expand(10)
            m = torch.randn(10, 1).type(tname).expand(10, 100)
            v = torch.randn(100).type(tname)
            res1 = torch.addmv(t, m, v)
            res2 = torch.zeros(10).type(tname)
           