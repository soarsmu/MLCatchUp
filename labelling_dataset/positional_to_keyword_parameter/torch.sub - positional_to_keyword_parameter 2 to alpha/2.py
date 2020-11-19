from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


class TorchSub(torch.nn.Module):
    def __init__(self):
        super(TorchSub, self).__init__()

    def forward(self, x, y):
        return torch.sub(x, y)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_torch_sub():
    return TorchSub()


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_rsub_int():
    return RSubInt()


class RSubFloat(torch.nn.Module):
    def __init__(self):
        super(RSubFloat, self).__init__()

    def forward(self, x):
        return 1.0 - x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_rsub_float():
    return RSubFloat()