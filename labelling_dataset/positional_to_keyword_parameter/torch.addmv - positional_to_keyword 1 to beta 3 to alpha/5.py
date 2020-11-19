import torch
import numpy as np

# Matrix + Matrix X vector
# Size 2
M = torch.randn(2)
mat = torch.randn(2, 3)
vec = torch.randn(3)
r = torch.addmv(M, mat, vec)

# Matrix x Matrix
# Size 2x4
mat1 = torch.randn(2, 3)
mat2 = torch.randn(3, 4)
r = torch.mm(mat1, mat2)

# Outer product of 2 vectors
# Size 3x2
v1 = torch.arange(1, 4)    # Size 3
v2 = torch.arange(1, 3)    # Size 2
r = torch.ger(v1, v2)

# Add M with outer product of 2 vectors
# Size 3x2
vec1 = torch.arange(1, 4)  # Size 3
vec2 = torch.arange(1, 3)  # Size 2
M = torch.zeros(3, 2)
r = torch.addr(M, vec1, vec2)

# Batch Matrix x Matrix
# Size 10x3x5
batch1 = torch.randn(10, 3, 4)
batch2 = torch.randn(10, 4, 5)
r = torch.bmm(batch1, batch2)

# Batch Matrix + Matrix x Matrix
# Performs a batch matrix-matrix product
# 3x4 + (5x3x4 X 5x4x2 ) -> 5x3x2
M = torch.randn(3, 2)
batch1 = torch.randn(5, 3, 4)
batch2 = torch.randn(5, 4, 2)
r = torch.addbmm(M, batch1, batch2)

# Move Tensors to GPU
if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    x + y

print(r)
