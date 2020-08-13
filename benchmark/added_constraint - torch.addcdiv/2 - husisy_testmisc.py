import numpy as np
import torch

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))


def test_torch_mm(N0=3, N1=5, N2=7):
    np0 = np.random.rand(N0, N1)
    np1 = np.random.rand(N1, N2)
    ret_ = np0 @ np1
    ret0 = torch.mm(torch.tensor(np0), torch.tensor(np1))
    assert hfe(ret_, ret0.numpy()) < 1e-5


def test_torch_addmm(N0=3, N1=5, N2=7):
    beta = np.random.randn()
    alpha = np.random.randn()
    np0 = np.random.rand(N0, N2)
    np1 = np.random.rand(N0, N1)
    np2 = np.random.randn(N1, N2)
    ret_ = beta*np0 + alpha*(np1 @ np2)
    ret0 = torch.addmm(torch.tensor(np0), torch.tensor(np1), torch.tensor(np2), beta=beta, alpha=alpha)
    assert hfe(ret_, ret0.numpy()) < 1e-5


def test_torch_multinomial():
    np0 = np.random.rand(5)
    np0 = np0 / np0.sum()
    num_sample = 100000
    tmp0 = torch.multinomial(torch.tensor(np0), num_sample, replacement=True).numpy()
    tmp2 = np.unique(tmp0, return_counts=True)[1]/num_sample
    assert hfe(np0, tmp2) < 0.01
    # TODO how to unittest replacement=False


def test_torch_addcdiv(N0=3):
    np0 = np.random.randn(N0)
    np1 = np.random.randn(N0)
    np2 = np.random.rand(N0) + 1
    np3 = np.random.randn()
    ret_ = np0 + np3 * np1 / np2
    ret0 = torch.addcdiv(torch.tensor(np0), torch.tensor(np1), torch.tensor(np2), value=np3)
    assert hfe(ret_, ret0.numpy()) < 1e-7


def test_torch_addcmul(N0=3):
    np0 = np.random.randn(N0)
    np1 = np.random.randn(N0)
    np2 = np.random.randn(N0)
    np3 = np.random.randn()
    ret_ = np0 + np3*np1*np2
    ret0 = torch.addcmul(torch.tensor(np0), torch.tensor(np1), torch.tensor(np2), value=np3)
    assert hfe(ret_, ret0.numpy()) < 1e-7


def test_torch_cupy_share_data():
    import torch.utils.dlpack
    import cupy as cp
    np0 = np.random.rand(3, 5)
    torch0 = torch.tensor(np0.copy()).to('cuda')

    cp0 = cp.fromDlpack(torch.utils.dlpack.to_dlpack(torch0))
    cp0[0,0] = 0.233
    assert hfe(torch0.to('cpu').numpy(), cp0.get()) < 1e-5

    cp1 = cp.array(np0)
    torch1 = torch.utils.dlpack.from_dlpack(cp1.toDlpack())
    torch1[0,0] = 0.233
    assert hfe(torch1.to('cpu').numpy(), cp1.get()) < 1e-5


def test_torch_memory_format():
    hf_shape_to_stride = lambda x: tuple(np.cumprod(np.asarray(x[1:])[::-1])[::-1].tolist() + [1])
    shape = (3,4,5,6)
    stride0 = hf_shape_to_stride(shape)
    tmp0 = hf_shape_to_stride((shape[0],shape[2],shape[3],shape[1]))
    stride1 = tmp0[0],tmp0[3],tmp0[1],tmp0[2]
    torch0 = torch.randn(*shape) #default torch.contiguous_format
    assert (torch0.stride()==stride0) and (tuple(torch0.shape)==shape)
    assert torch0.is_contiguous(memory_format=torch.contiguous_format)

    torch1 = torch0.contiguous(memory_format=torch.channels_last)
    assert (torch1.stride()==stride1) and (tuple(torch1.shape)==shape)
    assert torch1.is_contiguous(memory_format=torch.channels_last)

    torch2 = torch1.contiguous(memory_format=torch.contiguous_format)
    assert (torch2.stride()==stride0) and (tuple(torch2.shape)==shape)
    assert torch2.is_contiguous(memory_format=torch.contiguous_format)
