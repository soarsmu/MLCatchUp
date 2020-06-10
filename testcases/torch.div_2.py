import torch
def torch_div_test_deprecated():
    batch1 = torch.randn(5, 3, 4)
    batch2 = torch.randn(3, 4)
    print(torch.div(batch1, batch2))
    print(batch1 / batch2)
torch_div_test_deprecated()