import torch
def torch_div_test_deprecated():
    batch1 = torch.randint(3, 5, (4,4))
    batch2 = torch.randint(3, 5, (4,4))
    print(batch1 / 5)
    print(batch1)
    print(batch1.dtype)
torch_div_test_deprecated()