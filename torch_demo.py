import torch
import numpy as np
from torchvision import utils, models, datasets
from matplotlib import pyplot as plt

# print(torch.__version__)
# print(np.__version__)
# print("-" * 30)

# a = torch.rand((2, 3))
# print(a)
# print(a.shape)
# print(type(a))
# print("-" * 30)
#
# b = torch.tensor([3, 3, 3, 3, 2, 2, 2, 2, 2])
# print(b)
# print(b.shape)
# print(type(b))
# print(b.dtype)
# print("-" * 30)
#
# c = np.array([[2, 3], [3, 4]])
# print(c.shape)
# print(type(c))
# print(c.dtype)
# print("-" * 30)
#
# d = torch.tensor(c)
# dd = torch.from_numpy(c)
# print(d)
# print(d.shape)
# print(type(d))
# print(d.dtype)
# print(dd)
# print(dd.shape)
# print(type(dd))
# print(dd.dtype)
# print("-" * 30)
#
# # 零维张量 scalars标量
# e = torch.tensor(1)
# f = 1
# print(e)
# print(e.shape)
# print(type(e))
# print(type(f))
# print(e == f)
# print(e.item() == f)
# print("-" * 30)
#
# g = d.reshape(1, -1)
# h = d.flatten()
# print(g)
# print(h)
# print("-" * 30)
#
# i = torch.eye(3).double()
# j = torch.diag(h)
# print(i)
# print(j)
# print("-" * 30)

# dataset = datasets.MNIST('mnist', train=False, download=False)
# label = dataset.targets[:4]
# print(label)
# images = dataset.data[:4].float()
# # print(images)
# print(images.shape)
# print(images.size())
# print(images.numel())
# print(len(images))
# print("-" * 30)

# k = torch.rand([2, 3])
# l = torch.randn([2, 3])
# m = torch.normal(torch.tensor([2.0, 1.0, 3.0, 4.0]), torch.tensor([0.01, 0.02, 0.02, 0.01])).reshape(2, 2)
# n = torch.randint(1, 10, [2, 3])
# print(m)
# print("-" * 30)
#
# o = torch.randn(5)
# p = o.numpy()
# # q = np.array(o)
# r = o.tolist()
# s = [round(i, 2) for i in r]
# t = list(o)
# print(o)
# print(p)
# # print(q)
# print(r)
# print(s)
# print(t)


'''
make_grid
    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
            
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
            
        padding (int, optional): amount of padding. Default is 2.
        
'''

# Clipping input data to the valid range for imshow
# with RGB data ([0..1] for floats or [0..255] for integers).

# # 白色(255, 255, 255), 黑色(0, 0, 0)
# n = torch.rand(4, 3, 24, 24) * 255
# # print(n.shape)
# grid_n = utils.make_grid(n)
# # print(grid_n.shape)
# n_numpy = grid_n.numpy().astype(np.int32)
# # print(n_numpy.dtype)
# plt.imshow(np.transpose(n_numpy, (1, 2, 0)))
# plt.show()

# # 白色(255, 255, 255), 黑色(0, 0, 0)
# n = torch.rand(4, 3, 24, 24) * 1
# # print(n.shape)
# grid_n = utils.make_grid(n)
# # print(grid_n.shape)
# n_numpy = grid_n.numpy()
# # float32
# print(n_numpy.dtype)
# plt.imshow(np.transpose(n_numpy, (1, 2, 0)))
# plt.show()


