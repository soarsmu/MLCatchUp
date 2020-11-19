import os
import re
import yaml
import unittest
import textwrap
import torch
from collections import namedtuple


path = os.path.dirname(os.path.realpath(__file__))
aten_native_yaml = os.path.join(path, '../aten/src/ATen/native/native_functions.yaml')
all_operators_with_namedtuple_return = {
    'max', 'min', 'median', 'mode', 'kthvalue', 'svd', 'symeig', 'eig',
    'pstrf', 'qr', 'geqrf', 'solve', 'slogdet', 'sort', 'topk', 'gels',
    'triangular_solve'
}


class TestNamedTupleAPI(unittest.TestCase):

    def test_namedtuple_return(self):
        self.assertIs(ret.u, ret[0])
        self.assertIs(ret.pivot, ret[1])
        ret1 = torch.pstrf(b, out=tuple(ret))
        self.assertIs(ret1.u, ret1[0])
        self.assertIs(ret1.pivot, ret1[1])
        self.assertIs(ret1.u, ret[0])
        self.assertIs(ret1.pivot, ret[1])


if __name__ == '__main__':
    unittest.main()
