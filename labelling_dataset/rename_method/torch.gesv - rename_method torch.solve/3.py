import torch
from torch.autograd import Variable, grad
import torch.optim as optim
import time

def error_reproduce_process():
	n_data = 500
	ndim = 1000

	b = Variable(torch.randn(ndim, n_data))
	A = Variable(torch.randn(ndim, ndim))

	def gesv_wrapper(return_dict, i, *args):
		return_dict[i] = torch.gesv(*args)[0]

	return_dict = torch.multiprocessing.Manager().dict()

	return return_dict


def no_error_without_blas():
	n_data = 500
	ndim = 1000

	b = Variable(torch.randn(ndim, n_data))
	A = Variable(torch.randn(ndim, ndim))

	pool = torch.multiprocessing.Pool(1)
	res = torch.gesv(b, A)
	pool.close()
	pool.join()
	return res.get()

