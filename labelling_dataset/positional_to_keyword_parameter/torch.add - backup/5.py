from functools import wraps
import itertools
import unittest

import torch

from torch.testing._internal.common_utils import (TestCase, run_tests, load_tests,
                                                  TEST_NUMPY, torch_to_numpy_dtype_dict)
from torch.testing._internal.common_device_type import (instantiate_device_type_tests, onlyOnCPUAndCUDA,
                                                        dtypes, onlyCPU)


class TestTypePromotion(TestCase):
    @float_double_default_dtype
    def test_alternate_result(self, device):
        f = torch.tensor([1, 1, 1, 1], dtype=torch.float, device=device)
        o = torch.tensor([0, 0, 0, 0], dtype=torch.long, device=device)
        d = torch.tensor([1, 1, 1, 1], dtype=torch.double, device=device)
        torch.add(f, f, out=d)
        self.assertEqual(d.dtype, torch.double)
        # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
        self.assertEqualIgnoreType(f + f, d)
