from functools import wraps

import torch
import itertools

from torch.testing._internal.common_utils import TestCase, run_tests, load_tests
from torch.testing._internal.common_device_type import (instantiate_device_type_tests, onlyOnCPUAndCUDA,
                                                        dtypes)


class TestTypePromotion(TestCase):

    @float_double_default_dtype
    def test_alternate_result(self, device):
        f = torch.tensor([1, 1, 1, 1], dtype=torch.float, device=device)
        o = torch.tensor([0, 0, 0, 0], dtype=torch.long, device=device)
        d = torch.tensor([1, 1, 1, 1], dtype=torch.double, device=device)
        torch.add(f, f, out=d)
        self.assertEqual(d.dtype, torch.double)
        self.assertEqual(f + f, d)