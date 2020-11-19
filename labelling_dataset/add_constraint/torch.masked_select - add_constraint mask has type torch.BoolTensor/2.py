import numpy as np
import torch

# example 1
# torch.cat
# flag = True
flag = False
if flag:
    t = torch.ones((2, 3))
    # dim指的是维度,如果说0是行(即y轴),1是列(即x轴)
    t_0 = torch.cat([t, t], dim=0)
    t_1 = torch.cat([t, t], dim=1)

    print("t_0:{} \nshape:{}\nt_1:{}\n shape:{}".format(t_0, t_0.shape, t_1, t_1.shape))

# example 2
# torch.stack
# flag = True
flag = False
if flag:
    t = torch.ones((2, 3))
    # 这里用0维度,由于0维度已经被使用,所以他会再扩一个维度,变成三维张量,然后拼接
    t_stack = torch.stack([t, t], dim=0)
    # 注意两者效果不同,记得对比
    # t_stack = torch.stack([t, t], dim=2)
    print(t_stack)
    print(t_stack.shape)

# example 3
# torch.chunk
# flag = True
flag = False
if flag:
    a = torch.ones((2, 5))
    list_of_tensors = torch.chunk(a, dim=1, chunks=2)
    for idx, t in enumerate(list_of_tensors):
        print("第{}个张量:{}, shape is {}".format(idx+1, t, t.shape))


# example 4
# torch.split
# split不仅可以平均切分,还可以通过list指定每一份的数量
# flag = True
flag = False
if flag:
    a = torch.ones((2, 5))
    list_of_tensors = torch.split(a, [1, 2, 2],dim=1)
    for idx, t in enumerate(list_of_tensors):
        print("第{}个张量:{}, shape is {}".format(idx+1, t, t.shape))

# example 5
# torch.index_select
# flag = True
flag = False
if flag:
    t = torch.randint(0, 9, size=(3, 3))
    # 这里注意一定要torch.long类型
    idx = torch.tensor([0, 2], dtype=torch.long)
    t_select = torch.index_select(t, dim=0, index=idx)
    print(t, "\n", t_select)

# example 6
# torch.masked_select
# flag = True
flag = False
if flag:
    t = torch.randint(0, 9, size=(3, 3))
    # ge是对t中data的所有数据与5进行比较,ge是大于等于,大于等于括号中的数则为True
    # 另外还有gt,是小于等于
    mask = t.ge(5)
    # 返回的是一个一维张量
    t_select = torch.masked_select(t, mask=mask)
    print(t, "\n", t_select)


# example 7
# torch.reshape
# flag = True
flag = False
if flag:
    # 生成0-7的随机排列(8是指元素个数,也指定了值域)
    t = torch.randperm(8)
    t_reshape = torch.reshape(t, (2, 4))
    # 有时候可以用-1参数,让其自动识别
    # t_reshape = torch.reshape(t, (-1, 4))
    print("t:{}\nt_reshape:\n{}".format(t, t_reshape))
    # 他们的内存是共享的
    print(id(t.data))
    print(id(t_reshape.data))

# example 8
# torch.transpose
# flag = True
flag = False
if flag:
    t = torch.rand(2, 3, 4)
    # 交换维度
    t_transpose = torch.transpose(t, dim0=1, dim1=2)
    print(t.shape)
    print(t_transpose.shape)
    print(t)
    print(t_transpose)
    # 还可以使用torch.t(),可以作为二维张量的转置
    test = torch.rand((3,2))
    print(torch.t(test).shape)

# example 8
# torch.transpose & torch.unsqueeze
# flag = True
flag = False
if flag:
    t = torch.rand((1, 2, 3, 1))
    # 这个是压缩所有长度为1的维度
    t_sq = torch.squeeze(t)
    # 指定dim值,相当于对指定维度进行压缩
    t_0 = torch.squeeze(t, dim=0)
    t_1 = torch.squeeze(t, dim=1)
    print(t.shape)
    print(t_sq.shape)
    print(t_0.shape)
    print(t_1.shape)
    print(torch.unsqueeze(t_0, dim=0).shape)



# example 9
# torch.add
flag = True
# flag = False
if flag:
    t_0 = torch.randn((3, 3))
    t_1 = torch.ones_like(t_0)
    #注意中间是乘项因子
    t_add = torch.add(t_0, 10, t_1)
    print(t_0)
    print(t_add)