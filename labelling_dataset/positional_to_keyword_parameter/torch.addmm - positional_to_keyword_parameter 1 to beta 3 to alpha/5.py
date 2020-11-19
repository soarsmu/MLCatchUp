import torch

def addmm_test():
    mat2 = torch.randn(3, 3)
    res = torch.addmm(M, mat1, mat2)

    print(res)
