import unittest
from test.util import TVMTest
import torch
import torch_tvm


class TestCore(TVMTest):
   
    def test_fall_back(self, shape):
        inputs = torch.rand(shape)

        def add(input):
            return torch.add(input, 1, 2)

        torch.testing.assert_allclose(jit_out, tvm_out, rtol=0.01, atol=0.01)
