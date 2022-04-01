### torch基本操作

#### @ 

```
@ 和 * 代表矩阵的两种相乘方式：
@ 表示数学上定义的矩阵相乘；
* 表示两个矩阵对应位置处的元素相乘
```

```
a = torch.tensor([[2, 3], [3, 4]])
b = torch.tensor(([[2, 2, 1], [4, 3, 4]]))
print(a @ b)
print(torch.mm(a, b))
```

```
tensor([[16, 13, 14],
        [22, 18, 19]])
tensor([[16, 13, 14],
        [22, 18, 19]])
```

```
a = torch.tensor([[[2, 3], [3, 4]]])
b = torch.tensor(([[[2, 2, 1], [4, 3, 4]]]))
print(a @ b)
print(torch.bmm(a, b))
```

```
tensor([[[16, 13, 14],
         [22, 18, 19]]])
tensor([[[16, 13, 14],
         [22, 18, 19]]])
```

#### torch.mul() / *  

```
# * 等价于torch.mul()
a = torch.tensor([[[2., 2., 3.], [3., 2., 4.]]])
b = torch.tensor([[[5., 3., 4.], [2., 2., 3.]]])

print(a * b)
print(torch.mul(a, b))
print(a * 3)
print(torch.mul(a, 3))
```

```
tensor([[[10.,  6., 12.],
         [ 6.,  4., 12.]]])
tensor([[[10.,  6., 12.],
         [ 6.,  4., 12.]]])
tensor([[[ 6.,  6.,  9.],
         [ 9.,  6., 12.]]])
tensor([[[ 6.,  6.,  9.],
         [ 9.,  6., 12.]]])
```

#### torch.pow() / **

```
# ** 等价于torch.pow()
a = torch.tensor([[[2., 2., 3.], [3., 2., 4.]]])
b = torch.tensor([[[5., 3., 4.], [2., 2., 3.]]])

print(a ** b)
print(torch.pow(a, b))
print(a.pow(b))

print(a ** 3)
print(torch.pow(a, 3))
```

```
tensor([[[32.,  8., 81.],
         [ 9.,  4., 64.]]])
tensor([[[32.,  8., 81.],
         [ 9.,  4., 64.]]])
tensor([[[32.,  8., 81.],
         [ 9.,  4., 64.]]])
         
tensor([[[ 8.,  8., 27.],
         [27.,  8., 64.]]])
tensor([[[ 8.,  8., 27.],
         [27.,  8., 64.]]])
```

#### view()

```
view函数只能用在contiguous后的张量上，具体而言，就是在内存中连续存储的张量。
所以当tensor之前调用了transpose, permute等函数时，就会使tensor在内存中变得不再连续，就不能直接调用view函数。
所以，应该执行tensor.permute().contiguous().view()的操作

reshape() ≈ tensor.contiguous().view()
```

#### view_as()

```
# 返回与指定tensor相同size()的原tensor
a = torch.randn([2, 4])
b = torch.randn([4, 2])
c = b.view_as(a)
d = b.view(a.size())
print(b)
print(c)
print(d)
```

```
tensor([[ 0.5204, -0.1534],
        [-0.6361, -0.1641],
        [-0.9781,  1.0873],
        [-0.8863,  0.5894]])
tensor([[ 0.5204, -0.1534, -0.6361, -0.1641],
        [-0.9781,  1.0873, -0.8863,  0.5894]])
tensor([[ 0.5204, -0.1534, -0.6361, -0.1641],
        [-0.9781,  1.0873, -0.8863,  0.5894]])
```

#### reshape()

```
a = torch.tensor([[1, 2, 3], [2, 3, 1]])
b = torch.reshape(a, [3, -1])
print(b)
print(b.is_contiguous())
c = torch.reshape(torch.reshape(a, [3, -1]), [-1])
print(c)
print(c.is_contiguous())
```

```
tensor([[1, 2],
        [3, 2],
        [3, 1]])
True
tensor([1, 2, 3, 2, 3, 1])
True
```

```
d = torch.randn([2, 4])
print(d)
print(d.reshape([-1]))
print(d.reshape([-1]).size())
```

```
tensor([[ 0.1307,  0.0077,  1.2621, -1.1024],
        [-0.1706,  0.0334,  0.5554,  1.8288]])
        
tensor([ 0.1307,  0.0077,  1.2621, -1.1024, -0.1706,  0.0334,  0.5554,  1.8288])
torch.Size([8])
```

#### flatten()

```
d = torch.randn([2, 3])
print(d)
print(d.reshape([-1]))
print(d.reshape([-1]).is_contiguous())
print(d.flatten())
print(d.flatten().is_contiguous())
```

```
tensor([[-0.2054,  0.4836,  0.0598],
        [-0.7141, -0.3740,  0.1730]])
tensor([-0.2054,  0.4836,  0.0598, -0.7141, -0.3740,  0.1730])
True
tensor([-0.2054,  0.4836,  0.0598, -0.7141, -0.3740,  0.1730])
True
```

```
# tensor.flatten(start_dim, end_dim)

d = torch.randn([2, 2, 2])
print(d)
print(d.flatten(start_dim=1, end_dim=2))
```

```
tensor([[[-0.5437,  0.2121],
         [-0.2237,  0.8824]],

        [[ 1.4873, -0.4016],
         [ 0.5618, -0.0219]]])
tensor([[-0.5437,  0.2121, -0.2237,  0.8824],
        [ 1.4873, -0.4016,  0.5618, -0.0219]])
```

```
d = torch.randn([2, 2, 2])
print(d)
print(d.flatten(start_dim=0, end_dim=1))
```

```
tensor([[[-1.4989, -1.4792],
         [-0.9490,  1.4863]],

        [[ 1.7044,  0.1855],
         [-0.1624,  0.6382]]])
tensor([[-1.4989, -1.4792],
        [-0.9490,  1.4863],
        [ 1.7044,  0.1855],
        [-0.1624,  0.6382]])
```

#### expand() / expand_as()

```
expand()函数只能将size=1的维度扩展到更大的尺寸
cx = torch.tensor([[3], [2], [1]])
mx = torch.randn([3, 5])
cx.expand_as(mx) 等价于 cx.expand(mx.size())
```

```
# expand
dx = torch.tensor([[2, 3, 4]])
print(dx)
print(dx.expand([4, 3]))
ex = torch.tensor([[3], [2], [1]])
print(ex)
print(ex.expand(3, 4))
```

```
tensor([[2, 3, 4]])
tensor([[2, 3, 4],
        [2, 3, 4],
        [2, 3, 4],
        [2, 3, 4]])
tensor([[3],
        [2],
        [1]])
tensor([[3, 3, 3, 3],
        [2, 2, 2, 2],
        [1, 1, 1, 1]])
```

```
# expand_as
cx = torch.tensor([[3], [2], [1]])
mx = torch.randn([3, 5])
print(cx.expand_as(mx))
print(cx.expand(mx.size()))
```

```
tensor([[3, 3, 3, 3, 3],
        [2, 2, 2, 2, 2],
        [1, 1, 1, 1, 1]])
tensor([[3, 3, 3, 3, 3],
        [2, 2, 2, 2, 2],
        [1, 1, 1, 1, 1]])
```

#### squeeze()

```
pp = torch.tensor([[[1, 8, 3]], [[2, 6, 4]]])
print(pp)
print(pp.size())
print(pp.squeeze())
```

```
tensor([[[1, 8, 3]],

        [[2, 6, 4]]])
torch.Size([2, 1, 3])
tensor([[1, 8, 3],
        [2, 6, 4]])
```

```
# squeeze
ap = torch.randn([1, 3, 1, 2, 6])
print(ap.size())
print(ap.squeeze().size())
```

```
torch.Size([1, 3, 1, 2, 6])
torch.Size([3, 2, 6])
```

#### unsqueeze() / [None, :, None]

```
# unsqueeze and [None, :, None]
x = torch.tensor([1, 2, 3])
y1 = x.unsqueeze(1).unsqueeze(0)
y2 = x[None, :, None]
print(y1)
print(y2)
print(y1.size())
print(y2.size())
```

```
tensor([[[1],
         [2],
         [3]]])
tensor([[[1],
         [2],
         [3]]])
torch.Size([1, 3, 1])
torch.Size([1, 3, 1])
```

```
x = torch.from_numpy(np.array([[1, 2, 0.], [2, 3, 1.]])).long()
print(x)
y = x[:, :, None].expand(-1, -1, 4)
f = x.unsqueeze(2).expand(-1, -1, 4)
print(y)
print(f)
```

```
tensor([[1, 2, 0],
        [2, 3, 1]])
tensor([[[1, 1, 1, 1],
         [2, 2, 2, 2],
         [0, 0, 0, 0]],

        [[2, 2, 2, 2],
         [3, 3, 3, 3],
         [1, 1, 1, 1]]])
tensor([[[1, 1, 1, 1],
         [2, 2, 2, 2],
         [0, 0, 0, 0]],

        [[2, 2, 2, 2],
         [3, 3, 3, 3],
         [1, 1, 1, 1]]])
```

#### torch.t()

```
a = torch.tensor([1, 2]) # 1D
b = torch.tensor([[1, 2], [3, 4]]) # 2D
g = torch.rand([1, 2, 2]) # 3D
c = a.t()
d = b.t()
e = b.transpose(0, 1)
f = torch.t(b)
print(c)
print(d)
print(e)
print(f)
print(g.t())
```

```
tensor([1, 2])

tensor([[1, 3],
        [2, 4]])
tensor([[1, 3],
        [2, 4]])
tensor([[1, 3],
        [2, 4]])
        
# RuntimeError: t() expects a tensor with <= 2 dimensions, but self is 3D
print(g.t()) # 报错 
```

#### torch.eq()

```
a = torch.tensor([1, 2, 3, 2, 1])
b = torch.tensor([1, 3, 3, 1, 1])
c = torch.eq(a, b)
d = torch.eq(a, b).sum()
print(c)
print(d)
```

```
tensor([1, 0, 1, 0, 1], dtype=torch.uint8)
tensor(3)
```

```
a = torch.tensor([1, 2, 3, 2, 1]).view(5, -1)
b = torch.tensor([1, 3, 3, 1, 1]).view(5, -1)
c = torch.eq(a, b)
print(a)
print(c)
```

```
tensor([[1],
        [2],
        [3],
        [2],
        [1]])
tensor([[1],
        [0],
        [1],
        [0],
        [1]], dtype=torch.uint8)
```

```
# torch.eq(a, b)等价于(a == b)
a = torch.tensor([1, 0, 1, 1, 0, 0]).long()
b = torch.tensor([1, 0, 0, 1, 1, 0]).long()

print(a == b)
print(torch.eq(a, b))
print((a == b).sum().item())
print(torch.eq(a, b).sum().item())
```

```
tensor([1, 1, 0, 1, 0, 1], dtype=torch.uint8)
tensor([1, 1, 0, 1, 0, 1], dtype=torch.uint8)
4
4
```

#### tensor.transpose() / tensor.reshape() / tensor.permute()

```
c = torch.randn((2, 3))
print(c)
d = c.reshape((3, -1))
print(d)
print(d.is_contiguous())
f = c.transpose(-2, -1)
print(f)
print(f.is_contiguous())
n = c.permute((1, 0))
print(n)
print(n.is_contiguous())
```

```
tensor([[-1.0605, -0.2943, -1.0885],
        [-0.1894,  1.7407,  1.4679]])
tensor([[-1.0605, -0.2943],
        [-1.0885, -0.1894],
        [ 1.7407,  1.4679]])
True
tensor([[-1.0605, -0.1894],
        [-0.2943,  1.7407],
        [-1.0885,  1.4679]])
False
tensor([[-1.0605, -0.1894],
        [-0.2943,  1.7407],
        [-1.0885,  1.4679]])
False
```

#### torch.max() / torch.argmax()

```
import torch
import torch.nn as nn

torch.manual_seed(0)
# torch.cuda.manual_seed(0)

a = torch.rand([3, 5]) * 10
print(a)

# nn.Softmax(dim=-1)(a)等价于c = nn.Softmax(dim=-1), c(a)
b = nn.Softmax(dim=-1)(a)
c = nn.Softmax(dim=-1)
print(b)
print(c(a))
```

```
tensor([[4.9626, 7.6822, 0.8848, 1.3203, 3.0742],
        [6.3408, 4.9009, 8.9644, 4.5563, 6.3231],
        [3.4889, 4.0172, 0.2233, 1.6886, 2.9389]])
        
tensor([[0.0611, 0.9270, 0.0010, 0.0016, 0.0092],
        [0.0618, 0.0147, 0.8524, 0.0104, 0.0607],
        [0.2877, 0.4879, 0.0110, 0.0475, 0.1660]])
        
tensor([[0.0611, 0.9270, 0.0010, 0.0016, 0.0092],
        [0.0618, 0.0147, 0.8524, 0.0104, 0.0607],
        [0.2877, 0.4879, 0.0110, 0.0475, 0.1660]])
```

```
f = torch.max(b, dim=-1)
print(f)

print(f[0])
print(f[1])

print(f.values)
print(f.indices)
```

```
# print(f)
torch.return_types.max(
	values=tensor([0.9270, 0.8524, 0.4879]),
	indices=tensor([1, 2, 1])
)

# f[0]等价于f.values, f[1]等价于f.indices
# print(f[0]) print(f[1])
tensor([0.9270, 0.8524, 0.4879])
tensor([1, 2, 1])

# print(f.values) print(f.indices)
tensor([0.9270, 0.8524, 0.4879])
tensor([1, 2, 1])
```

```
# b.argmax(dim=-1)等价于torch.argmax(b, dim=-1)
e = torch.argmax(b, dim=-1)
print(e)

target = torch.LongTensor([1, 2, 0])  # label

equal = torch.eq(e, target)
count = torch.eq(e, target).sum()
equal2 = (b.argmax(dim=-1) == target)
count2 = (b.argmax(dim=-1) == target).sum()
print(equal)
print(count)
print(equal2)
print(count2)
```

```
# torch.argmax(b, dim=-1) 返回最大概率值的索引位置
tensor([1, 2, 1])

# torch.eq(e, target)
tensor([ True,  True, False])

# torch.eq(e, target).sum()
tensor(2)

# (b.argmax(dim=-1) == target)
tensor([ True,  True, False])

# (b.argmax(dim=-1) == target).sum()
tensor(2)
```

#### torch.mm()

```
# 矩阵乘法，即两个二维tensor做矩阵乘法
a = torch.tensor([[1, 2], [2, 2]])
b = torch.tensor([[1, 2], [3, 2]])
print(a)
print(b)
print(torch.mm(a, b))
```

```
tensor([[1, 2],
        [2, 2]])
tensor([[1, 2],
        [3, 2]])
tensor([[7, 6],
        [8, 8]])
```

```
# 传入三维张量做矩阵乘法会报错，必须二维：
a = torch.tensor([[[1, 2], [2, 2]]])
b = torch.tensor([[[1, 2], [3, 2]]])
print(a)
print(b)
print(torch.mm(a, b))

RuntimeError: matrices expected, got 3D, 3D tensors
```

#### torch.bmm()

```
# torch.bmm
c = torch.randn((2, 3))
d = c.reshape((3, -1))
e = torch.bmm(c, d)
print(e)
报错：bmm传入的必须是三维张量
RuntimeError: Expected 3-dimensional tensor, but got 2-dimensional tensor for argument #1 'batch1'
```

```
# torch.bmm()：矩阵乘法，tensor必须是三维，并且后两维参与矩阵乘法运算。
c = torch.tensor([[2, 1, 1], [3, 2, 4]]).view(-1, 2, 3).expand(2, -1, -1)
print(c)
d = torch.tensor([[2, 1, 1], [3, 2, 4]]).reshape(3, -1).view(-1, 3, 2).expand(2, -1, -1)
print(d)
e = torch.bmm(c, d)
print(e)

print(c.size())
print(d.size())
print(e.size())
```

```
tensor([[[2, 1, 1],
         [3, 2, 4]],

        [[2, 1, 1],
         [3, 2, 4]]])
tensor([[[2, 1],
         [1, 3],
         [2, 4]],

        [[2, 1],
         [1, 3],
         [2, 4]]])
tensor([[[ 7,  9],
         [16, 25]],

        [[ 7,  9],
         [16, 25]]])
         
torch.Size([2, 2, 3])
torch.Size([2, 3, 2])
torch.Size([2, 2, 2])
```

```
# torch.bmm(c, d).float().sigmoid().log() # sigmoid不支持long数据类型，所以要加.float()
c = torch.tensor([[[2, 1, 1]], [[2, 1, 1]]])
d = torch.tensor([[[2], [-2], [-1]], [[2], [-2], [-1]]])
print(torch.bmm(c, d))

out_loss_sigmoid = torch.bmm(c, d).float().sigmoid()
print(out_loss_sigmoid)
out_loss_logsigmoid = torch.bmm(c, d).float().sigmoid().log()
print(out_loss_logsigmoid)
out_loss_squeeze = out_loss_sigmoid.squeeze()
print(out_loss_squeeze)
```

```
# torch.bmm(c, d)
tensor([[[1]],

        [[1]]])
# torch.bmm(c, d).float().sigmoid()        
tensor([[[0.7311]],

        [[0.7311]]])
# torch.bmm(c, d).float().sigmoid().log()        
tensor([[[-0.3133]],

        [[-0.3133]]])
# out_loss_sigmoid.squeeze()        
tensor([0.7311, 0.7311])
```

#### torch.matmul()

```
# torch.mm()仅仅是供矩阵相乘使用，使用范围较为狭窄。 torch.matmul()使用的场景就比较多了
```

```
# 两个一维向量matmul，返回dot product向量点乘结果：
a = torch.tensor([1, 2, 3])
b = torch.tensor([1, 2, 4])
c = torch.matmul(a, b)
print(c)

tensor(17)
```

```
# 两个二维矩阵matmul，做矩阵乘法，等价于torch.mm()
a = torch.tensor([[1, 2], [3, 5]])
b = torch.tensor([[1, 3], [4, 2]])
c = torch.matmul(a, b)
print(c)

tensor([[ 9,  7],
        [23, 19]])
```

```
# 第一个参数是一维的，第二个参数是二维的，则第一个参数先增加一维，做矩阵乘法，最后把结果去掉一维
a = torch.tensor([1, 2])
b = torch.tensor([[1, 3, 4], [4, 2, 5]])
c = torch.matmul(a, b)
print(c)

tensor([ 9,  7, 14])
```

```
# 两个高维张量做的是最后两维上的矩阵乘法，高维必须对齐，高维维度不统一会自动根据broadcasting机制补齐。
# 如果无法补齐并对齐，则会报错
a = torch.randn(2, 3, 4, 2)
b = torch.randn(2, 3, 2, 5)
c = torch.randn(3, 2, 5)
print(torch.matmul(a, b).shape)
print(torch.matmul(a, c).size())
```

```
torch.Size([2, 3, 4, 5])
torch.Size([2, 3, 4, 5])
```

```
# 下面这个例子就没办法通过broadcasting机制对齐a和c，所以会报错。
a = torch.randn(2, 3, 4, 2)
c = torch.randn(2, 2, 5)
print(torch.matmul(a, c).size())

RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 1

a = torch.randn(2, 3, 4, 2)
c = torch.randn(1, 2, 5)
print(torch.matmul(a, c).size())

out:
torch.Size([2, 3, 4, 5])
```

#### torch.cat()

```
a = torch.tensor([[1, 2, 3]])
b = torch.cat((a, a, a), dim=0)
c = torch.cat((a, a, a), dim=1)
print(b)
print(c)
```

```
tensor([[1, 2, 3],
        [1, 2, 3],
        [1, 2, 3]])
tensor([[1, 2, 3, 1, 2, 3, 1, 2, 3]])
```

#### torch.chunk()

```
# chunk()操作后返回一个元组
x = torch.randn([1, 2, 9])
qkv = torch.chunk(x, chunks=3, dim=-1)
print(qkv)
print(type(qkv))

q, k, v = torch.chunk(x, chunks=3, dim=-1)
print(q)
print(type(q))
```

```
(tensor([[[-0.0785, -0.2077, -0.3348],
          [ 1.6771,  0.3143, -1.0334]]]), 
 tensor([[[-0.8349, -1.3555, -1.6933],
          [-1.0923, -1.6441, -1.4329]]]), 
 tensor([[[ 0.4299, -0.6353,  0.7164],
          [-1.6127, -0.1418,  0.8005]]]))
<class 'tuple'>

tensor([[[-0.0785, -0.2077, -0.3348],
         [ 1.6771,  0.3143, -1.0334]]])
<class 'torch.Tensor'>
```

#### torch.gather()

```
# torch.gather(input(tensor), dim(int), index(LongTensor))
inputs = torch.randn(3, 5)
print(inputs)
indexs = torch.tensor([[1], [2], [0]])
print(indexs)
out = torch.gather(inputs, dim=1, index=indexs)
print(out)
```

```
tensor([[ 1.0350,  1.5290, -0.2648,  0.9630,  0.6335],
        [-1.4681, -1.6431, -2.6290,  0.0174,  0.7508],
        [ 0.7789, -0.9383,  1.1640,  0.5291, -0.1891]])
# index
tensor([[1],
        [2],
        [0]])
# out.shape = index.shape        
tensor([[ 1.5290],
        [-2.6290],
        [ 0.7789]])
```

```
# 用在交换句子先后顺序处
# 如：交换句子先后顺序为2, 3, 1, 4
inp = [
    [2, 3, 4, 5, 0, 0],
    [1, 4, 3, 0, 0, 0],
    [4, 2, 2, 5, 7, 0],
    [1, 0, 0, 0, 0, 0]
]
inp = torch.tensor(inp)
print(inp)
# indexs作为索引值需要减1，即第一个句子的索引值为0.
index = torch.tensor([2, 3, 1, 4]).unsqueeze(1).expand(-1, inp.size(1)) - 1
print(index)
out = torch.gather(inp, dim=0, index=index)
print(out)
```

```
tensor([[2, 3, 4, 5, 0, 0],
        [1, 4, 3, 0, 0, 0],
        [4, 2, 2, 5, 7, 0],
        [1, 0, 0, 0, 0, 0]])
        
tensor([[1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2],
        [0, 0, 0, 0, 0, 0],
        [3, 3, 3, 3, 3, 3]])
        
tensor([[1, 4, 3, 0, 0, 0],
        [4, 2, 2, 5, 7, 0],
        [2, 3, 4, 5, 0, 0],
        [1, 0, 0, 0, 0, 0]])
```

#### torch.roll()

```
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
```

```
# shifts大于0向右下移，小于0向左上移
a = torch.arange(1, 17).view(4, 4)
print(a)
print(torch.roll(a, shifts=1, dims=0))
print(torch.roll(a, shifts=1, dims=1))
```

```
tensor([[ 1,  2,  3,  4],
        [ 5,  6,  7,  8],
        [ 9, 10, 11, 12],
        [13, 14, 15, 16]])
tensor([[13, 14, 15, 16],
        [ 1,  2,  3,  4],
        [ 5,  6,  7,  8],
        [ 9, 10, 11, 12]])
tensor([[ 4,  1,  2,  3],
        [ 8,  5,  6,  7],
        [12,  9, 10, 11],
        [16, 13, 14, 15]])
```

```
a = torch.arange(1, 17).view(4, 4)
print(a)
print(torch.roll(a, shifts=-1, dims=0))
print(torch.roll(a, shifts=-1, dims=1))
```

```
tensor([[ 1,  2,  3,  4],
        [ 5,  6,  7,  8],
        [ 9, 10, 11, 12],
        [13, 14, 15, 16]])
tensor([[ 5,  6,  7,  8],
        [ 9, 10, 11, 12],
        [13, 14, 15, 16],
        [ 1,  2,  3,  4]])
tensor([[ 2,  3,  4,  1],
        [ 6,  7,  8,  5],
        [10, 11, 12,  9],
        [14, 15, 16, 13]])
```

```
a = torch.arange(1, 17).view(4, 4)
print(a)
print(torch.roll(a, shifts=(-1, 1), dims=(0, 1)))
```

```
tensor([[ 1,  2,  3,  4],
        [ 5,  6,  7,  8],
        [ 9, 10, 11, 12],
        [13, 14, 15, 16]])
tensor([[ 8,  5,  6,  7],
        [12,  9, 10, 11],
        [16, 13, 14, 15],
        [ 4,  1,  2,  3]])
```

```
# torch.roll()处理图片：

img_path = r"F:\2022lianxi\Transformer\pictures\tupian\5.png"
img_PIL = Image.open(img_path)
print(img_PIL)

PIL_to_tensor = transforms.ToTensor()  # 必须要实例化
tensor_to_PIL = transforms.ToPILImage()  # convert tensor to PIL image
img_Tensor = PIL_to_tensor(img_PIL).unsqueeze(0)
print(img_Tensor.size())


def imshow(tensor, title=None):
    plt.figure(figsize=(10, 8))
    image = tensor.clone().squeeze()  # remove the fake batch dimension
    image = tensor_to_PIL(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.savefig(f"./{title}.png")
    plt.show()


imshow(img_Tensor, title='original')
y = torch.roll(img_Tensor, shifts=-70, dims=3)
z = torch.roll(img_Tensor, shifts=70, dims=2)
w = torch.roll(img_Tensor, shifts=(70, 70), dims=(2, 3))
imshow(y, title='roll_left_right')
imshow(z, title='roll_up_down')
imshow(w, title="roll_all")
```

<img src="F:\学习资料\数据分析\original.png" style="zoom: 50%;" />

<img src="F:\学习资料\数据分析\roll_left_right.png" style="zoom: 50%;" />

<img src="F:\学习资料\数据分析\roll_up_down.png" style="zoom: 50%;" />

<img src="F:\学习资料\数据分析\roll_all.png" style="zoom:50%;" />

#### torch.numel()

```
# 返回tensor中元素的总个数。
a = torch.randn([5, 5])
print(a.size())
print(torch.numel(a))
```

```
torch.Size([5, 5])
25
```

#### torch.arange() / torch.range() 

```
# torch.arange() 推荐使用
# torch.range() 会报UserWarning，被弃用
a = torch.arange(start=0, end=5, step=1)
print(a)
print(a.dtype)

b = torch.range(start=0, end=5, step=1)
print(b)
print(b.dtype)
```

```
tensor([0, 1, 2, 3, 4])
torch.int64
tensor([0., 1., 2., 3., 4., 5.])
torch.float32

# torch.range()
UserWarning: torch.range is deprecated in favor of torch.arange and will be removed in 0.5. Note that arange generates values in [start; end), not [start; end].
```

#### torch.linspace()

```
c = torch.linspace(0, 5, 8)
d = torch.linspace(start=0, end=10, steps=11)
print(c)
print(d)
print(d.dtype)
```

```
tensor([0.0000, 0.7143, 1.4286, 2.1429, 2.8571, 3.5714, 4.2857, 5.0000])
tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])
torch.float32
```

#### torch.from_numpy()

```
# 假设a的最后一列代表label，需要通过切片操作分割：
a = np.array([[1, 2, 3, 0], [2, 3, 1, 1], [3, 5, 2, 0]], dtype=np.float32)
print(a.shape)

b = torch.from_numpy(a[:, :-1])
c = torch.from_numpy(a[:, -1])
d = torch.from_numpy(a[:, [-1]])
print(b)
print(c)
print(d)
```

```
# a.shape
(3, 4)

# b = torch.from_numpy(a[:, :-1])
tensor([[1., 2., 3.],
        [2., 3., 1.],
        [3., 5., 2.]])
        
# c = torch.from_numpy(a[:, -1])
tensor([0., 1., 0.])

# d = torch.from_numpy(a[:, [-1]])
tensor([[0.],
        [1.],
        [0.]])
```

#### tensor.item()

```
a = torch.tensor([1, 2, 3, 2, 1])
print(a.item())
报错：ValueError: only one element tensors can be converted to Python scalars
```

```
b = torch.tensor(5)
print(b)
print(b.item())
print(type(b))
print(type(b.item()))
```

```
tensor(5)
5
<class 'torch.Tensor'>
<class 'int'>
```

#### tensor.detach().cpu().numpy().tolist()

```
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
y = y.to(device)
print(y)
print(y.data)
print(y.cpu().data)
```

```
tensor([[3., 3.],
        [3., 3.]], device='cuda:0', grad_fn=<CopyBackwards>)
tensor([[3., 3.],
        [3., 3.]], device='cuda:0')
tensor([[3., 3.],
        [3., 3.]])
```

```
print(y.device)
print(y.type())
print(y.requires_grad)
print(y.grad_fn)
```

```
# y.device
cuda:0
# y.type()
torch.cuda.FloatTensor
# y.requires_grad
True
# y.grad_fn
<CopyBackwards object at 0x0000020A9D8D8488>
```

```
# y.data等价于y.detach()!!!
print(y.data)
print(y.detach())
print(y.detach().cpu())
```

```
tensor([[3., 3.],
        [3., 3.]], device='cuda:0')
tensor([[3., 3.],
        [3., 3.]], device='cuda:0')
tensor([[3., 3.],
        [3., 3.]])
```

```
print(y.detach().cpu().device)
print(y.detach().requires_grad)
print(y.detach().grad_fn)
```

```
cpu
False
None
```

#### contiguous() and is_contiguous()

```
# contiguous() and is_contiguous()
a = torch.randint(1, 4, [3, 4])
print(a)
print(a.is_contiguous())
b = a.transpose(-2, -1)
print(b)
print(b.is_contiguous())
c = a.transpose(-2, -1).contiguous()
print(c)
print(c.is_contiguous())
```

```
tensor([[1, 2, 1, 2],
        [3, 2, 1, 1],
        [1, 1, 1, 1]])
True
tensor([[1, 3, 1],
        [2, 2, 1],
        [1, 1, 1],
        [2, 1, 1]])
False
tensor([[1, 3, 1],
        [2, 2, 1],
        [1, 1, 1],
        [2, 1, 1]])
True
```

```
# Call .contiguous() before .view()
a = torch.randint(1, 4, [3, 4])
b = a.transpose(-2, -1) # 不连续
print(b.is_contiguous())

d = b.contiguous().view(-1)
print(d)
```

```
False
tensor([1, 3, 2, 1, 2, 3, 3, 1, 1, 2, 3, 3])
```

#### torch.set_default_tensor_type

```
# 指定默认的tensor类型为GPU上的(FloatTensor)
a = torch.ones(2, 3, requires_grad=True).double()
torch.set_default_tensor_type('torch.cuda.FloatTensor')
b = torch.ones(2, 3, requires_grad=True)
print(a.dtype)
print(a.type())
print(a.is_cuda)
print(b.dtype)
print(b.type())
print(b.is_cuda)
```

```
torch.float64
torch.DoubleTensor
False
torch.float32
torch.cuda.FloatTensor
True
```

#### with torch.no_grad()

```
# with torch.no_grad()
x = torch.randn(5, requires_grad=True)
y = torch.randn(5, requires_grad=True)
z = torch.randn(5, requires_grad=True)
w = z * (x + y)
print(w)
print(w.requires_grad)
print(w.grad_fn)

with torch.no_grad():
    n = z * (x + y)
print(n.requires_grad)
print(n.grad_fn)
```

```
tensor([ 0.2328, -0.9288,  0.2339,  0.4743,  0.2850], grad_fn=<MulBackward0>)
True
<MulBackward0 object at 0x000001AB8C16F9C8>
False
None
```

#### keepdim参数的用法

```
a = torch.tensor([1, 2, 3, 4, 5])
a = a.expand([2, 5])
d = a.mean(dim=-1)
f = a.mean(dim=-1, keepdim=True)
print(d)
print(f)
a.dtype: torch.int64
报错：RuntimeError: Can only calculate the mean of floating types. Got Long instead.
```

```
a = torch.tensor([1, 2, 3, 4, 5])
a = a.expand([2, 5]).float()
print(a.dtype)
b = a.data.max(dim=-1)
c = a.data.max(dim=-1, keepdim=True)
print(b)
print(c)
```

```
torch.float32
torch.return_types.max(values=tensor([5., 5.]), indices=tensor([4, 4]))
torch.return_types.max(values=tensor([[5.], [5.]]), indices=tensor([[4], [4]]))
```

```
a = torch.tensor([1, 2, 3, 4, 5])
a = a.expand([2, 5]).float()
print(a.dtype)
d = a.mean(dim=-1)
f = a.mean(dim=-1, keepdim=True)
print(d)
print(f)
```

```
torch.float32
tensor([3., 3.])
tensor([[3.],
        [3.]])
```

```
target = torch.tensor([1, 0, 1, 2, 1]) # 标签
output = torch.tensor([2, 1, 3]).expand(5, 3).float() # 网络输出，batchsize=5，3分类问题
pred = output.max(dim=-1, keepdim=True)[1]  # 获取最大值索引
print(pred)
print(target.view_as(pred))
print(torch.eq(pred, target.view_as(pred)))
print(pred.eq(target.view_as(pred)).sum())
print(pred.eq(target.view_as(pred)).sum().item())
```

```
tensor([[2],
        [2],
        [2],
        [2],
        [2]])
tensor([[1],
        [0],
        [1],
        [2],
        [1]])
tensor([[0],
        [0],
        [0],
        [1],
        [0]], dtype=torch.uint8)
tensor(1)
1
```

#### torch.manual_seed(0) / torch.cuda.manual_seed(0)

```
import torch

# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# 固定随机数种子
```



### torch数据类型

#### numpy、tensor默认的浮点数数据类型

```
# numpy and tensor默认的浮点数数据类型
hh = np.array([3.2, 33.4])  # float64
jj = torch.from_numpy(hh)
print(hh.dtype)
print(jj.dtype)

kk = torch.tensor([3.4, 3.22])  # torch.float32
ll = kk.numpy()
print(kk.dtype)
print(ll.dtype)
```

```
float64
torch.float64
torch.float32
float32
```

#### numpy、tensor默认的整数型数据类型

```
a = torch.tensor([1, 2, 3])
print(a.dtype)
b = np.array([1, 2, 3])
print(b.dtype)
```

```
torch.int64
int32
```

#### torch.tensor(0) 和 (int) 0 的区别

```
print(type(torch.tensor(0.2)))
print(type(0.2))

print(torch.tensor(10) + 9)
print(torch.tensor(10).item() + 9)

print(type(torch.tensor(10) + 9))
print((torch.tensor(10) + 9).dtype)
```

```
<class 'torch.Tensor'>
<class 'float'>
tensor(19)
19
<class 'torch.Tensor'>
torch.int64
```

#### torch.tensor() 和 torch.Tensor() 区别

```
# torch.Tensor是torch.FloatTensor的简写：
a = torch.tensor([1, 2, 3])
b = torch.Tensor([1, 2, 3])
c = torch.FloatTensor([1, 2, 3])
print(a.dtype)
print(b.dtype)
print(c.dtype)
```

```
torch.int64
torch.float32
torch.float32
```



### torch数据读取、数据处理

#### torch.utils.data.TensorDataset

```
from torch.utils.data import TensorDataset, DataLoader, Dataset
a = torch.randn(2, 3, 3)
b = torch.randn(2)
c = TensorDataset(a, b)

print(type(c))
print(c.tensors)
print(c.tensors[0])
print(c.tensors[1])
```

```
<class 'torch.utils.data.dataset.TensorDataset'>
# c.tensors返回一个元组(datas, labels)
(tensor([[[ 1.0926,  1.0053,  0.7203],
         [ 0.4553,  1.6796, -0.1436],
         [-0.1075, -0.2964, -0.7254]],

        [[ 0.2967, -1.1462,  0.5565],
         [-0.4346, -0.6789,  1.1471],
         [-0.0343,  0.6797, -0.1790]]]), tensor([0.8307, 0.6307]))
         
         
tensor([[[ 1.0926,  1.0053,  0.7203],
         [ 0.4553,  1.6796, -0.1436],
         [-0.1075, -0.2964, -0.7254]],

        [[ 0.2967, -1.1462,  0.5565],
         [-0.4346, -0.6789,  1.1471],
         [-0.0343,  0.6797, -0.1790]]])
         
         
tensor([0.8307, 0.6307])
```

#### torch.utils.data.DataLoader

```
from torch.utils.data import TensorDataset, DataLoader, Dataset
a = torch.randn(10, 3, 3)
b = torch.randn(10)
c = TensorDataset(a, b)
d = DataLoader(dataset=c, batch_size=2, shuffle=True)
f = DataLoader(dataset=c, batch_size=2, shuffle=False)

print(len(c))
print(len(d.dataset))

print(type(c))
print(type(d))

print(d.batch_size)
print(d.batch_sampler.sampler)
print(f.batch_sampler.sampler)
```

```
print(len(c))
print(len(d.dataset))
10
10

print(type(c))
print(type(d))
<class 'torch.utils.data.dataset.TensorDataset'>
<class 'torch.utils.data.dataloader.DataLoader'>

print(d.batch_size)
2

print(d.batch_sampler.sampler)
print(f.batch_sampler.sampler)
<torch.utils.data.sampler.RandomSampler object at 0x0000020BFD0EC748>
<torch.utils.data.sampler.SequentialSampler object at 0x0000020BFD0EC808>
```

#### torch.utils.data.Dataset



### torch深度学习

#### model.parameters()

```
model = nn.Sequential(
    nn.Linear(2, 3),
    nn.ReLU(),
    nn.Linear(3, 4),
)

for param in model.parameters():
    print(param.size())
    print(param)
```

```
torch.Size([3, 2])
Parameter containing:
tensor([[0.5055, 0.4364],
        [0.2350, 0.2113],
        [0.5361, 0.1507]], requires_grad=True)
        
torch.Size([3])
Parameter containing:
tensor([-0.1848, -0.6061,  0.5522], requires_grad=True)

torch.Size([4, 3])
Parameter containing:
tensor([[ 0.5022,  0.0603, -0.3823],
        [ 0.1500,  0.5260, -0.0833],
        [ 0.0825,  0.0783,  0.0445],
        [-0.4067,  0.3245, -0.1826]], requires_grad=True)
        
torch.Size([4])
Parameter containing:
tensor([ 0.1289, -0.4655,  0.0439, -0.3490], requires_grad=True
```

#### model.named_parameters()

```
model = nn.Sequential(
    nn.Linear(2, 3),
    nn.ReLU(),
    nn.Linear(3, 5),
)
print(model)

for name, param in model.named_parameters():
    print(name)
    print(param)
```

```
Sequential(
  (0): Linear(in_features=2, out_features=3, bias=True)
  (1): ReLU()
  (2): Linear(in_features=3, out_features=5, bias=True)
)

0.weight
Parameter containing:
tensor([[-0.2659,  0.6019],
        [-0.1569, -0.5109],
        [-0.3207,  0.5985]], requires_grad=True)
0.bias
Parameter containing:
tensor([ 0.4060, -0.3293, -0.5593], requires_grad=True)

2.weight
Parameter containing:
tensor([[-0.5138, -0.1465, -0.0619],
        [ 0.3738,  0.2652, -0.5151],
        [-0.2655,  0.5333, -0.5660],
        [ 0.5204, -0.4184, -0.5215],
        [ 0.0285, -0.3088, -0.4685]], requires_grad=True)
2.bias
Parameter containing:
tensor([-0.5734,  0.5450, -0.5270,  0.2654, -0.2589], requires_grad=True)
```

#### model.named_children():

```
# model.children() / model.named_children()

model2 = nn.Sequential(
    collections.OrderedDict([
        ('conv1', nn.Conv2d(1, 2, 3)),
        ('relu1', nn.ReLU()),
        ('conv2', nn.Conv2d(2, 3, 3)),
        ('relu2', nn.ReLU())
    ])
)
print(model2)

for name, model in model2.named_children():
    if name in ["conv1", "conv2"]:
        print(model)
```

```
Sequential(
  (conv1): Conv2d(1, 2, kernel_size=(3, 3), stride=(1, 1))
  (relu1): ReLU()
  (conv2): Conv2d(2, 3, kernel_size=(3, 3), stride=(1, 1))
  (relu2): ReLU()
)

Conv2d(1, 2, kernel_size=(3, 3), stride=(1, 1))
Conv2d(2, 3, kernel_size=(3, 3), stride=(1, 1))
```

#### model.named_modules()

```
# model.named_modules() / model.modules()

import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm):
        super(Net, self).__init__()
        self.layer1 = nn.ModuleList([nn.Linear(2, 3), nn.Linear(3, 4)])
        self.layer2 = nn.Sequential(
            norm_layer(5),
            nn.Conv2d(3, 3, (3, 3))
        )

    def forward(self, x):
        pass


model = Net()

for name, m in model.named_modules():
    print(name, m)
```

```
 Net(
  (layer1): ModuleList(
    (0): Linear(in_features=2, out_features=3, bias=True)
    (1): Linear(in_features=3, out_features=4, bias=True)
  )
  (layer2): Sequential(
    (0): LayerNorm((5,), eps=1e-05, elementwise_affine=True)
    (1): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
  )
)

layer1 ModuleList(
  (0): Linear(in_features=2, out_features=3, bias=True)
  (1): Linear(in_features=3, out_features=4, bias=True)
)

layer1.0 Linear(in_features=2, out_features=3, bias=True)
layer1.1 Linear(in_features=3, out_features=4, bias=True)

layer2 Sequential(
  (0): LayerNorm((5,), eps=1e-05, elementwise_affine=True)
  (1): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
)

layer2.0 LayerNorm((5,), eps=1e-05, elementwise_affine=True)
layer2.1 Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
```

```
for name, m in model.named_children():
    print(name, m)
```

```
layer1 ModuleList(
  (0): Linear(in_features=2, out_features=3, bias=True)
  (1): Linear(in_features=3, out_features=4, bias=True)
)
layer2 Sequential(
  (0): LayerNorm((5,), eps=1e-05, elementwise_affine=True)
  (1): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
)
```

#### model.children()和model.modules()的区别

```
net = nn.Sequential(nn.Linear(2, 3),
                    nn.ReLU(),
                    nn.Sequential(nn.Sigmoid(), nn.ReLU()))

for module in net.children():
    print(module)

for module in net.modules():
    print(module)
```

```
# model.children() 
Linear(in_features=2, out_features=3, bias=True)
ReLU()
Sequential(
  (0): Sigmoid()
  (1): ReLU()
)

# model.modules()
Sequential(
  (0): Linear(in_features=2, out_features=3, bias=True)
  (1): ReLU()
  (2): Sequential(
    (0): Sigmoid()
    (1): ReLU()
  )
)
Linear(in_features=2, out_features=3, bias=True)
ReLU()
Sequential(
  (0): Sigmoid()
  (1): ReLU()
)
Sigmoid()
ReLU()
```

#### model.modules()注意事项：

```
# 重复的模块只返回一次！！！
module = nn.Linear(2, 2)
net1 = nn.Sequential(module, module)
net2 = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))

for idx, m in enumerate(net1.modules()):
    print(idx, '->', m)

for idx, m in enumerate(net2.modules()):
    print(idx, '->', m)
```

```
0 -> Sequential(
  (0): Linear(in_features=2, out_features=2, bias=True)
  (1): Linear(in_features=2, out_features=2, bias=True)
)
1 -> Linear(in_features=2, out_features=2, bias=True)

0 -> Sequential(
  (0): Linear(in_features=2, out_features=2, bias=True)
  (1): Linear(in_features=2, out_features=2, bias=True)
)
1 -> Linear(in_features=2, out_features=2, bias=True)
2 -> Linear(in_features=2, out_features=2, bias=True)
```

#### model.state_dict()

```
model = nn.Sequential(
    nn.Linear(2, 3),
    nn.Linear(3, 3)
)

for param in model.parameters():
    print(param)

torch.save(model.state_dict(), "model_weights.pth")
```

```
Parameter containing:
tensor([[-0.6907,  0.2325],
        [-0.1513, -0.4461],
        [ 0.1588, -0.4333]], requires_grad=True)
        
Parameter containing:
tensor([ 0.4660, -0.5134,  0.2211], requires_grad=True)

Parameter containing:
tensor([[-0.5509,  0.4608,  0.0832],
        [-0.1472,  0.5695, -0.4624],
        [ 0.5758,  0.4314,  0.3451]], requires_grad=True)
        
Parameter containing:
tensor([-0.4105, -0.4634,  0.3261], requires_grad=True)
```

```
model = nn.Sequential(
    nn.Linear(1, 2),
    nn.Linear(2, 3),
    nn.BatchNorm2d(3)
)
print(model.state_dict())
print(model.state_dict()['0.weight'])
```

```
# 返回一个OrderedDict有序字典
OrderedDict([('0.weight', tensor([[-0.8391], [ 0.1679]])), ('0.bias', tensor([-0.2238, -0.1801])), ('1.weight', tensor([[-0.2319, -0.4267],
                     [-0.3257, -0.3234],
                     [-0.2448,  0.1819]])), ('1.bias', tensor([-0.3275,  0.4113, -0.2962])), 
('2.weight', tensor([0.4807, 0.1162, 0.4520])), ('2.bias', tensor([0., 0., 0.])), 
('2.running_mean', tensor([0., 0., 0.])), ('2.running_var', tensor([1., 1., 1.])), 
('2.num_batches_tracked', tensor(0))])

# model.state_dict()['0.weight']
tensor([[-0.8391], 
	    [ 0.1679]])
```

#### model.load_state_dict()

```
model = nn.Sequential(
    nn.Linear(2, 3),
    nn.Linear(3, 3)
)

model.load_state_dict(torch.load(r"./model_weights.pth"))

for param in model.parameters():
    print(param)
```

```
Parameter containing:
tensor([[-0.6907,  0.2325],
        [-0.1513, -0.4461],
        [ 0.1588, -0.4333]], requires_grad=True)
        
Parameter containing:
tensor([ 0.4660, -0.5134,  0.2211], requires_grad=True)

Parameter containing:
tensor([[-0.5509,  0.4608,  0.0832],
        [-0.1472,  0.5695, -0.4624],
        [ 0.5758,  0.4314,  0.3451]], requires_grad=True)
        
Parameter containing:
tensor([-0.4105, -0.4634,  0.3261], requires_grad=True)
```

#### initialize_parameters / _init_weights

```
# isinstance() 
a = 2
print(isinstance(a, int))
print(isinstance(a, str))
```

```
True
False
```

```
print(isinstance([2, 3], (int, str, list, tuple)))
print(isinstance({2, 3}, (int, str, list, tuple)))
print(isinstance({2, 3}, (int, str, list, tuple, set)))
print(isinstance({2: 3, 4: 10}, (int, str, list, tuple, set)))
print(isinstance({2: 3, 4: 10}, (int, str, list, tuple, set, dict)))
```

```
True
False
True
False
True
```

```
# isinstance()考虑继承关系，type()不考虑继承关系

class A:
    pass


class B(A):
    pass


print(isinstance(A(), A))
print(type(A()) == A)

print(isinstance(B(), A))
print(type(B()) == A)

print(type(A()))
print(type(B()))
```

```
True
True
True
False
<class '__main__.A'>
<class '__main__.B'>
```

```
linear = nn.Linear(3, 6)
print(type(linear))
print(isinstance(linear, nn.Linear))
```

```
<class 'torch.nn.modules.linear.Linear'>
True
```

```
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm):
        super(Net, self).__init__()
        self.module_list = nn.ModuleList([nn.Linear(2, 3), nn.Linear(3, 4), nn.Linear(4, 5)])
        self.norm_layer = norm_layer(5)
        self.initialize_parameters()

    def initialize_parameters(self):
        self.apply(self._init_weights)

    # initialize nn.Linear and nn.LayerNorm
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, 3.)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 2.)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 5.)
            nn.init.constant_(m.bias, 0.5)

    def forward(self, x):
        pass


model = Net()

for name, params in model.named_parameters():
    print(name, params)
```

```
module_list.0.weight Parameter containing:
tensor([[3., 3.],
        [3., 3.],
        [3., 3.]], requires_grad=True)
        
module_list.0.bias Parameter containing:
tensor([2., 2., 2.], requires_grad=True)

module_list.1.weight Parameter containing:
tensor([[3., 3., 3.],
        [3., 3., 3.],
        [3., 3., 3.],
        [3., 3., 3.]], requires_grad=True)
        
module_list.1.bias Parameter containing:
tensor([2., 2., 2., 2.], requires_grad=True)

module_list.2.weight Parameter containing:
tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]], requires_grad=True)
        
module_list.2.bias Parameter containing:
tensor([2., 2., 2., 2., 2.], requires_grad=True)

norm_layer.weight Parameter containing:
tensor([5., 5., 5., 5., 5.], requires_grad=True)

norm_layer.bias Parameter containing:
tensor([0.5000, 0.5000, 0.5000, 0.5000, 0.5000], requires_grad=True)
```

#### requires_grad / requires_grad_  / grad_fn

```
a = torch.tensor([1, 2]).float()
print(a)
print(a.requires_grad)
a.requires_grad_(True)
print(a)
print(a.requires_grad)
```

```
tensor([1., 2.])
False
tensor([1., 2.], requires_grad=True)
True
```

```
# tensor.detach()作用等价于tensor.data
x = torch.tensor([1., 2.])
x.requires_grad_(True)
print(x)
print(x.requires_grad)
print(x.data)
print(x.data.requires_grad)

y = x + 2
print(y.grad_fn)
print(y.data.grad_fn)
```

```
tensor([1., 2.], requires_grad=True)
True
tensor([1., 2.])
False

<AddBackward0 object at 0x000001BDFDE669D0>
None
```

#### Softmax LogSoftmax NLLLoss CrossEntropyLoss

```
# NLLLoss Softmax LogSoftmax CrossEntropyLoss
torch.manual_seed(0)

sm = nn.Softmax(dim=-1)
lsm = nn.LogSoftmax(dim=-1)
nloss = nn.NLLLoss()
cel = nn.CrossEntropyLoss()

x = torch.randn(2, 4)
labels = torch.tensor([1, 3])
# Softmax + torch.log() = LogSoftmax
y = sm(x)
z = torch.log(y)
w = lsm(x)
print(x)
print(y)
print(z)
print(w)
```

```
tensor([[ 1.5410, -0.2934, -2.1788,  0.5684],
        [-1.0845, -1.3986,  0.4033,  0.8380]])
tensor([[0.6402, 0.1022, 0.0155, 0.2421],
        [0.0769, 0.0562, 0.3407, 0.5262]])
tensor([[-0.4460, -2.2804, -4.1658, -1.4186],
        [-2.5647, -2.8788, -1.0768, -0.6421]])
tensor([[-0.4460, -2.2804, -4.1658, -1.4186],
        [-2.5647, -2.8788, -1.0768, -0.6421]])
```

```
# Softmax + Log + NLLLoss = CrossEntropyLoss
print(nloss(z, labels))
print(cel(x, labels))
```

```
tensor(1.4613)
tensor(1.4613)
```

#### F.nll_loss and F.log_softmax(x, dim=-1)

```
import torch.nn.functional as F

torch.manual_seed(0)

sm = nn.Softmax(dim=-1)
lsm = nn.LogSoftmax(dim=-1)
nloss = nn.NLLLoss()
cel = nn.CrossEntropyLoss()

x = torch.randn(2, 4)
labels = torch.tensor([1, 3])
```

```
# 以下四行等价
print(F.nll_loss(F.log_softmax(x, dim=-1), labels))
print(nloss(torch.log(sm(x)), labels))
print(nloss(lsm(x), labels))
print(cel(x, labels))
```

```
tensor(1.4613)
tensor(1.4613)
tensor(1.4613)
tensor(1.4613)
```

#### torch.nn.MSELoss(reduction="mean")

```
# torch.nn.MSELoss(reduction="mean")
# torch.nn.MSELoss(reduction="sum")
# reduction默认为mean

a = torch.tensor([1, 2, 3, 2, 1, 2, 3]).float()
b = torch.tensor([2, 3, 3, 1, 4, 2, 1]).float()
mseloss = nn.MSELoss()
mseloss_sum = nn.MSELoss(reduction="sum")
mseloss_mean = nn.MSELoss(reduction="mean")
c = mseloss(a, b)
d = mseloss_sum(a, b)
e = mseloss_mean(a, b)
```

```
print(c)
print(d)
print(e)
print(torch.pow(a - b, 2).sum())
print(torch.pow(a - b, 2).sum() / a.size(0))
```

```
tensor(2.2857)
tensor(16.)
tensor(2.2857)
tensor(16.)
tensor(2.2857)
```

#### nn.ReLU()

```
x = torch.randn([3, 5])
y = nn.ReLU()(x)
print(x)
print(y)
print(x.dtype)
print(y.dtype)
```

```
tensor([[ 0.4706,  1.3639, -1.1335, -1.7092, -1.1925],
        [-1.0446,  1.0443,  0.6022,  0.7273,  0.6503],
        [ 0.7332, -1.5993, -1.0839, -0.1172, -1.6932]])
        
tensor([[0.4706, 1.3639, 0.0000, 0.0000, 0.0000],
        [0.0000, 1.0443, 0.6022, 0.7273, 0.6503],
        [0.7332, 0.0000, 0.0000, 0.0000, 0.0000]])
        
torch.float32
torch.float32
```

#### nn.ModuleList()

```
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, depth):
        super(Net, self).__init__()
        self.list = [nn.Linear(3, 4), nn.ReLU()]
        self.module_one = nn.ModuleList(
            [nn.Conv2d(3, 5, kernel_size=(3, 3)), nn.ReLU(), nn.Linear(3, 4, bias=False)])
        self.module_two = nn.ModuleList([nn.Linear(3, 3) for _ in range(depth)])

    def forward(self):
        pass


model = Net(3)
print(model)

for name, params in model.named_parameters():
    print(name, params.size())
```

```
Net(
  (module_one): ModuleList(
    (0): Conv2d(3, 5, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): Linear(in_features=3, out_features=4, bias=False)
  )
  (module_two): ModuleList(
    (0): Linear(in_features=3, out_features=3, bias=True)
    (1): Linear(in_features=3, out_features=3, bias=True)
    (2): Linear(in_features=3, out_features=3, bias=True)
  )
)
```

```
module_one.0.weight torch.Size([5, 3, 3, 3])
module_one.0.bias torch.Size([5])
# 没有module_one.2.bias，因为bias被设置为了False 
module_one.2.weight torch.Size([4, 3])

module_two.0.weight torch.Size([3, 3])
module_two.0.bias torch.Size([3])

module_two.1.weight torch.Size([3, 3])
module_two.1.bias torch.Size([3])

module_two.2.weight torch.Size([3, 3])
module_two.2.bias torch.Size([3])
```

```
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.module_list = nn.ModuleList([nn.Linear(3, 4), nn.Linear(4, 5), nn.Linear(5, 6)])

    def forward(self, x):
        for i, module in enumerate(self.module_list):
            print(self.module_list[i])
            print(module)
            x = module(x)
        return x


model = Net()
ins = torch.randn([2, 3])
output = model(ins)
print(output.size())
```

```
Linear(in_features=3, out_features=4, bias=True)
Linear(in_features=3, out_features=4, bias=True)
Linear(in_features=4, out_features=5, bias=True)
Linear(in_features=4, out_features=5, bias=True)
Linear(in_features=5, out_features=6, bias=True)
Linear(in_features=5, out_features=6, bias=True)

torch.Size([2, 6])
```

#### nn.Sequential()

##### nn.Sequential(*layers)

```
# nn.Sequential(*layers)
layers = []
depth = 3
for _ in range(depth):
    linear = nn.Linear(in_features=3, out_features=3)
    layers.append(linear)
block = nn.Sequential(layers)
报错：TypeError: list is not a Module subclass
```

```
layers = []
depth = 3
for _ in range(depth):
    linear = nn.Linear(in_features=3, out_features=3)
    layers.append(linear)
block = nn.Sequential(*layers)
print(block)
```

```
Sequential(
  (0): Linear(in_features=3, out_features=3, bias=True)
  (1): Linear(in_features=3, out_features=3, bias=True)
  (2): Linear(in_features=3, out_features=3, bias=True)
)
```

#### nn.Parameter()

```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 3)
        self.linear1.weight = nn.Parameter(torch.arange(1, 5).view(2, -1).float())
        self.linear1.bias = nn.Parameter(torch.ones(2))
        self.linear2.weight = nn.Parameter(torch.arange(1, 7).view(3, -1).float())
        self.linear2.bias = nn.Parameter(torch.ones(3))
        # self.linear2.bias = torch.ones(3)
        self.v = nn.Parameter(torch.arange(1, 4).view(3, -1).float())
        # self.v = torch.arange(1, 4).view(3, -1).float()

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return torch.matmul(x, self.v)


x = torch.tensor([[1., 1], [1., 1]], requires_grad=True)
net = Net()
y = net(x)
print(y)
for name, params in net.named_parameters():
    print(name, params.size())
```

```
tensor([[318.],
        [318.]], grad_fn=<MmBackward>)
        
v  torch.Size([3, 1])
linear1.weight torch.Size([2, 2])
linear1.bias torch.Size([2])
linear2.weight torch.Size([3, 2])
linear2.bias torch.Size([3])
```

```
# torch.ones(3)替换nn.Parameter(torch.ones(3))会报错：
# self.linear2.bias = nn.Parameter(torch.ones(3))
self.linear2.bias = torch.ones(3)
```

```
TypeError: cannot assign 'torch.FloatTensor' as parameter 'bias' (torch.nn.Parameter or None expected)
```

```
# self.v 如果不是nn.Parameter(tensor)，而仅仅是tensor，则不属于网络参数(net.parameters)，后续不能进行值的更新
# 即model.parameters() 和 model.named_parameters()中找不到self.v了

# self.v = nn.Parameter(torch.arange(1, 4).view(3, -1).float())
self.v = torch.arange(1, 4).view(3, -1).float()
```

```
x = torch.tensor([[1., 1], [1., 1]], requires_grad=True)
net = Net()
y = net(x)
print(y)

for name, params in net.named_parameters():
    print(name, params.size())
```

```
tensor([[318.],
        [318.]], grad_fn=<MmBackward>)
        
linear1.weight torch.Size([2, 2])
linear1.bias torch.Size([2])
linear2.weight torch.Size([3, 2])
linear2.bias torch.Size([3])
```

#### nn.ParameterDict()

```
# Holds parameters in a dictionary.
# 注意！！ Pytorch低版本(1.1.0)的nn.ParameterDict()有bug，会报错，需要提升Pytorch版本(使用的是1.11.0)

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.params = nn.ParameterDict({
            'left': nn.Parameter(torch.randn(2, 10)),
            'right': nn.Parameter(torch.randn(3, 10))
        })

    def forward(self, x, choice):
        x = self.params[choice].mm(x)
        return x


x = torch.randn(10, 5)
model = MyModule()
print(model(x, "left").size())
print(model(x, 'right').size())

print(torch.__version__)
```

```
torch.Size([2, 5])
torch.Size([3, 5])

1.11.0
```

#### nn.Unfold() / nn.Fold()

```
x = torch.randn((1, 3, 224, 224))

unfold = nn.Unfold(kernel_size=16, stride=16)
out = unfold(x)
print(out.shape)

fold = nn.Fold(output_size=(224, 224), kernel_size=16, stride=16)
fold_out = fold(out)
print(fold_out.shape)
```

```
torch.Size([1, 768, 196])
torch.Size([1, 3, 224, 224])
```

```
# nn.Unfold具体数据的变化：

x = torch.randn((1, 2, 4, 4))
print(x)
unfold = nn.Unfold(kernel_size=(2, 2), stride=2)
out = unfold(x)
print(out)
print(out.shape)
```

```
tensor([[[[-0.4076,  2.5362, -0.7620, -1.7236],
          [-0.2419, -0.7838,  1.6163,  1.6338],
          [ 1.2798, -0.9674,  1.7931, -1.8931],
          [-0.9873, -2.0545, -1.0363,  0.8650]],

         [[-1.0302,  2.3535, -1.5835,  0.8423],
          [ 1.9641, -0.4432, -0.3730, -0.1483],
          [ 0.3178, -0.3085,  1.2643,  0.6429],
          [ 0.1096,  0.8262, -0.4802, -0.8150]]]])
          
tensor([[[-0.4076, -0.7620,  1.2798,  1.7931],
         [ 2.5362, -1.7236, -0.9674, -1.8931],
         [-0.2419,  1.6163, -0.9873, -1.0363],
         [-0.7838,  1.6338, -2.0545,  0.8650],
         [-1.0302, -1.5835,  0.3178,  1.2643],
         [ 2.3535,  0.8423, -0.3085,  0.6429],
         [ 1.9641, -0.3730,  0.1096, -0.4802],
         [-0.4432, -0.1483,  0.8262, -0.8150]]])
         
torch.Size([1, 8, 4])
```

#### nn.Embedding()

```
# 自定义初始化权重
# weight.size [3,4] 3表示整个词表里一共有三个词，4表示每个词的embedding_dim = 4
weight = torch.tensor([[1, 2.3, 3, 2], [4, 5.1, 6.3, 1.8], [2, 3.2, 1, 9.2]]).type(torch.float32)
embedding = nn.Embedding.from_pretrained(weight)
print(embedding.weight)
print(embedding.embedding_dim)
```

```
Parameter containing:
tensor([[1.0000, 2.3000, 3.0000, 2.0000],
        [4.0000, 5.1000, 6.3000, 1.8000],
        [2.0000, 3.2000, 1.0000, 9.2000]])
4
```

```
# batch_index.size [2,2] 第一个2表示一个batch有两个句子，第二个2表示每个句子有两个词
# batch_index中的数字 1,2 and 2,0 代表的是索引值
# embedding(batch_index)：根据batch_index中的索引值取出词表中对应索引位置的词向量。
# 可以理解为首先：word2index，然后：index2vector
batch_index = torch.tensor([[1, 2], [2, 0]]).long()
res = embedding(batch_index)
print(res)
print(res.size())
```

```
tensor([[[4.0000, 5.1000, 6.3000, 1.8000],
         [2.0000, 3.2000, 1.0000, 9.2000]],

        [[2.0000, 3.2000, 1.0000, 9.2000],
         [1.0000, 2.3000, 3.0000, 2.0000]]])
torch.Size([2, 2, 4])
```

```
# 参数padding_idx的作用：

embed = nn.Embedding(8, 4, padding_idx=0)  # padding_idx默认是0
print(embed.weight)
```

```
Parameter containing:
tensor([[ 0.0000,  0.0000,  0.0000,  0.0000],
        [-0.2991, -0.1028, -1.4979, -0.7255],
        [-0.8350, -0.1324, -0.2675, -0.2429],
        [ 0.8115,  0.3323, -0.0085, -1.2204],
        [ 0.2850, -0.2272, -0.8456,  1.3180],
        [ 0.7470,  1.6380,  0.0634, -0.4141],
        [-0.9003,  0.1889, -0.3519,  0.2983],
        [ 0.8059,  0.1323,  0.5408,  0.4539]], requires_grad=True)
        
空白填充 "PAD"对应的数字索引是0，则词表的第0行也应该设置为全零向量，即 padding_idx=0。      
```

```
import torch
import torch.nn as nn

torch.manual_seed(0)

n, d, m = 3, 5, 7
embedding = nn.Embedding(n, d)
W = torch.randn((m, d), requires_grad=True)
idx = torch.LongTensor([1, 2])

a = embedding.weight.clone() @ W.t()
b = embedding(idx) @ W.t()

print(embedding.weight)
print(W.size())
print(W.t().size())
print(a)
print(b)
```

```
Parameter containing:
tensor([[ 1.5410, -0.2934, -2.1788,  0.5684, -1.0845],
        [-1.3986,  0.4033,  0.8380, -0.7193, -0.4033],
        [-0.5966,  0.1820, -0.8567,  1.1006, -1.0712]], requires_grad=True)
        
torch.Size([7, 5])
torch.Size([5, 7])

tensor([[-0.2170, -1.8936,  1.3007, -4.9483, -3.4463,  1.9805,  0.1348],
        [ 0.1020,  0.2249, -0.1989,  1.3998,  1.8563, -2.3244,  1.9884],
        [ 0.5963, -1.2898,  3.8417, -0.1438,  0.5073, -2.8721,  0.8256]],
       grad_fn=<MmBackward0>)
       
tensor([[ 0.1020,  0.2249, -0.1989,  1.3998,  1.8563, -2.3244,  1.9884],
        [ 0.5963, -1.2898,  3.8417, -0.1438,  0.5073, -2.8721,  0.8256]],
       grad_fn=<MmBackward0>)

Process finished with exit code 0
```

#### F.one_hot(tensor, num_classes)

```
import torch.nn.functional as F

x = torch.tensor([[[0, 0], [1, 2]]])  # [B,H,W]
print(x.size())

y = torch.nn.functional.one_hot(x, num_classes=3)  # [B,H,W,C]
print(y)
print(y.size())

z = y.permute(0, 3, 1, 2)  # [B,C,H,W]
print(z)
print(z.size())
```

```
# 三分类的one-hot编码：
# x.size()
torch.Size([1, 2, 2])

tensor([[[[1, 0, 0],
          [1, 0, 0]],

         [[0, 1, 0],
          [0, 0, 1]]]])
       
torch.Size([1, 2, 2, 3])  # [B,H,W,C]   

tensor([[[[1, 1],
          [0, 0]],

         [[0, 0],
          [1, 0]],

         [[0, 0],
          [0, 1]]]])

torch.Size([1, 3, 2, 2])  # [B,C,H,W]
```

#### F.interpolate()

```
# 3D张量插值：
x = torch.tensor([[[1., 2, 3, 4]]])
y = torch.tensor([[[2., 3, 1, 2, 9, 3, 2, 8]]])
print(y.size())
x = F.interpolate(x, size=y.size(2))
print(x)
```

```
torch.Size([1, 1, 8])
tensor([[[1., 1., 2., 2., 3., 3., 4., 4.]]])
```

```
x = torch.tensor([[[1., 2, 3, 4, 6, 5, 7, 2]]])
z = F.interpolate(x, scale_factor=0.5)
n = F.interpolate(x, size=4)
print(z)
print(n)
print(x.shape)
print(z.shape)
```

```
tensor([[[1., 3., 6., 7.]]])
tensor([[[1., 3., 6., 7.]]])
torch.Size([1, 1, 8])
torch.Size([1, 1, 4])
```

```
# 4D张量插值：
# mode默认值：mode='nearest'
x = torch.tensor([[[[1., 2], [3, 4]]]])
a = F.interpolate(x, scale_factor=2, mode='nearest')
b = F.interpolate(x, size=(4, 4), mode='nearest')
c = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
d = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
print(x)
print(a)
print(b)
print(c)
print(d)
```

```
tensor([[[[1., 2.],
          [3., 4.]]]])
          
# size=(4, 4)：相当于把原x尺寸放大了两倍，即 scale_factor=2，两者在这里等价         
tensor([[[[1., 1., 2., 2.],
          [1., 1., 2., 2.],
          [3., 3., 4., 4.],
          [3., 3., 4., 4.]]]])
          
tensor([[[[1., 1., 2., 2.],
          [1., 1., 2., 2.],
          [3., 3., 4., 4.],
          [3., 3., 4., 4.]]]])
          
# align_corners=True         
tensor([[[[1.0000, 1.3333, 1.6667, 2.0000],
          [1.6667, 2.0000, 2.3333, 2.6667],
          [2.3333, 2.6667, 3.0000, 3.3333],
          [3.0000, 3.3333, 3.6667, 4.0000]]]])
          
# align_corners=False          
tensor([[[[1.0000, 1.2500, 1.7500, 2.0000],
          [1.5000, 1.7500, 2.2500, 2.5000],
          [2.5000, 2.7500, 3.2500, 3.5000],
          [3.0000, 3.2500, 3.7500, 4.0000]]]])
```

#### nn.Conv2d

```
nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1, dilation=1, groups=1)
```

##### dilation=1, groups=1

```
# DW卷积和groups参数有关，空洞卷积和dilation参数有关
# dilation = 1
a = torch.randn([5, 5])
b = a[0:3, 0:3]
print(a)
print(b)
```

```
tensor([[ 0.4905, -0.9557,  1.2325, -1.1359, -0.2415],
        [-0.2208, -0.3955, -1.1970,  0.4977, -2.5628],
        [-1.6920,  0.2495, -1.3728, -1.5495,  0.0813],
        [-1.3096, -1.7708, -0.5739,  0.2058,  0.8447],
        [ 0.5366, -0.8759,  0.3627,  0.3022,  0.3646]])
tensor([[ 0.4905, -0.9557,  1.2325],
        [-0.2208, -0.3955, -1.1970],
        [-1.6920,  0.2495, -1.3728]])
```

```
# dilation = 2
a = torch.randn([5, 5])
b = a[0:5:2, 0:5:2]
print(a)
print(b)
```

```
tensor([[ 0.4139, -0.9559,  0.5818, -1.8564,  0.2059],
        [-0.1770,  0.9327,  1.4006,  0.3202,  0.1709],
        [-1.2150, -0.6472,  0.2158, -0.6667, -1.4109],
        [ 0.1433,  1.0594,  0.9888, -2.3837,  0.0054],
        [ 0.1621, -0.2979,  0.6693, -0.5151,  0.8845]])
tensor([[ 0.4139,  0.5818,  0.2059],
        [-1.2150,  0.2158, -1.4109],
        [ 0.1621,  0.6693,  0.8845]])
```

#### 使用Conv1d替换Linear层，实现相同的功能

```python
# MLP: Multi-layer perceptron(多层感知机)
in_channels = 3
out_channels = 5
mlplinear = torch.nn.Linear(in_channels, out_channels)
mlpconv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1)
print(mlpconv.weight.size())
print(mlpconv.bias.size())
print(mlplinear.weight.size())
print(mlplinear.bias.size())
```

```
torch.Size([5, 3, 1])
torch.Size([5])
torch.Size([5, 3])
torch.Size([5])
```

```
# 用同一组初始化的weight和bias初始化两个模型的参数
ww = torch.arange(1, 16).reshape([5, 3]).float()
bb = torch.randn(5)
mlpconv.weight = torch.nn.Parameter(ww.unsqueeze(-1))  # torch.Size([5, 3, 1])
mlpconv.bias = torch.nn.Parameter(bb)
mlplinear.weight = torch.nn.Parameter(ww)
mlplinear.bias = torch.nn.Parameter(bb)
print(mlpconv.bias)
print(mlpconv.weight)
print(mlplinear.weight)
```

```
Parameter containing:
tensor([ 1.5837, -1.6966, -0.2335, -0.4583,  0.7336], requires_grad=True)

Parameter containing:
tensor([[[ 1.],
         [ 2.],
         [ 3.]],

        [[ 4.],
         [ 5.],
         [ 6.]],

        [[ 7.],
         [ 8.],
         [ 9.]],

        [[10.],
         [11.],
         [12.]],

        [[13.],
         [14.],
         [15.]]], requires_grad=True)
         
Parameter containing:
tensor([[ 1.,  2.,  3.],
        [ 4.,  5.,  6.],
        [ 7.,  8.,  9.],
        [10., 11., 12.],
        [13., 14., 15.]], requires_grad=True)
```

```python
# 数据输入，计算输出
input_x = torch.arange(1, 13).reshape(4, -1).unsqueeze(0).float()
res_conv = mlpconv(input_x.transpose(1, 2)).transpose(1, 2)
print(res_conv)
print(res_conv.size())
res_linear = mlplinear(input_x)
print(res_linear)
print(res_linear.size())
```

```
tensor([[[ 15.5837,  30.3034,  49.7665,  67.5417,  86.7336],
         [ 33.5837,  75.3034, 121.7665, 166.5417, 212.7336],
         [ 51.5837, 120.3034, 193.7665, 265.5417, 338.7336],
         [ 69.5837, 165.3034, 265.7665, 364.5417, 464.7336]]],
       grad_fn=<TransposeBackward0>)
       
torch.Size([1, 4, 5])

tensor([[[ 15.5837,  30.3034,  49.7665,  67.5417,  86.7336],
         [ 33.5837,  75.3034, 121.7665, 166.5417, 212.7336],
         [ 51.5837, 120.3034, 193.7665, 265.5417, 338.7336],
         [ 69.5837, 165.3034, 265.7665, 364.5417, 464.7336]]],
       grad_fn=<AddBackward0>)
       
torch.Size([1, 4, 5])
```

#### forward()的原理

```
# 先补充 *args **kwargs __call__()相关知识
# *把函数test()接收到的多个参数1, 3, "aa", "sd"打包成了元组(1, 3, 'aa', 'sd')，赋值给了形参args
# **把函数test()接收到的多个键值对参数k='zxf', age=23打包成了字典{'k': 'zxf', 'age': 23}，赋值给了形参kwargs
def test(*args, **kwargs):
    print(args)
    print(kwargs)

test(1, 3, "aa", "sd", k='zxf', age=23)
```

```
(1, 3, 'aa', 'sd')
{'k': 'zxf', 'age': 23}
```

```
# __call__()用法
# r(arg1, arg2, ...)只是 x.__call__(arg1, arg2, ...)的简写
class Role:
    def __init__(self, name):
        self.name = name

    # 定义__call__方法
    def __call__(self):
        print('执行Role对象')


r = Role('管理员')
# 直接调用Role对象，就是调用该对象的__call__方法
r()
r.__call__()
```

```
执行Role对象
执行Role对象
```

```
# 前向传播forward()原理

class Demo(object):
    def __init__(self):
        pass

    def __call__(self, *a, **b):
        print(a)
        print(b)
        res1 = self.forward(a)
        res2 = self.forward(b)
        return res1, res2

    def forward(self, x):
        return x


demo = Demo()
demo.__call__(1, 2, 3, k=4, m=5)
res = demo(1, 2, 3, k=4, m=5)
print(res)
```

```
# demo.__call__(1, 2, 3, k=4, m=5)
(1, 2, 3)
{'k': 4, 'm': 5}
# demo(1, 2, 3, k=4, m=5)
(1, 2, 3)
{'k': 4, 'm': 5}
# res
((1, 2, 3), {'k': 4, 'm': 5})
```

#### nn.Linear()实现一个简单的线性回归模型

```
import torch
import torch.nn as nn

x_data = torch.arange(1, 11).reshape(10, -1).float()
y_data = torch.arange(3, 31, 3).float().view_as(x_data) + 1


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out


model = LinearModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(1500):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    if (epoch + 1) % 100 == 0:
        print(epoch, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

```
# loss
99 0.06601104140281677
199 0.01714356429874897
299 0.0032414584420621395
399 0.00040989098488353193
499 3.503045081743039e-05
599 2.017015049204929e-06
699 7.717756034253398e-08
799 1.922853654079404e-09
899 4.476987530699361e-11
999 7.457856381065664e-12
1099 7.457856381065664e-12
1199 7.457856381065664e-12
1299 7.457856381065664e-12
1399 7.457856381065664e-12
1499 7.457856381065664e-12
```

```
# print("w:", model.linear.weight.item())
# print("b", model.linear.bias.item())
w: 2.9999990463256836
b 1.0000056028366089
```

```
x_test = torch.tensor([[109.], [47.]])
y_test = model(x_test)
print("data:", x_test)
# y_test.detach().data.numpy()
print("predict:", y_test)
```

```
data tensor([[109.],
             [ 47.]])
predict: tensor([[327.9999],
        		 [142.0000]], grad_fn=<AddmmBackward0>)
```

#### class PatchEmbedding(nn.Module):

```
import torch
import torch.nn as nn
from timm.models.layers import to_2tuple
```

```
class PatchEmbedding(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        print("image_size:", img_size)
        print("patch_size:", patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        print("proj.size:", self.proj.weight.size())
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        # [b,c,h,w] -> [b, embed_dim, img_size / patch_size, img_size / patch_size]
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> [B, embed_dim, num_patches] -> [B, num_patches, embed_dim]
        x = self.norm(x)
        return x
```

```
class Net(nn.Module):
    def __init__(self, image_size, patch_size, in_chans, embed_dim):
        super(Net, self).__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_chans, embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        return x
```

```
net = Net(image_size=224, patch_size=16, in_chans=3, embed_dim=96)
ins = torch.randn([3, 3, 224, 224])
out = net(ins)
print("out.size", out.size())
```

```
image_size: (224, 224)
patch_size: (16, 16)
proj.size: torch.Size([96, 3, 16, 16])
out.size torch.Size([3, 196, 96])
```

#### Pytorch网络结构及参数打印

```
import torch
import torch.nn as nn
from torchsummary import summary


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(5, 10, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(160, 80),
            nn.Linear(80, 10)
        )

    def forward(self, x):
        x = self.model(x)
        x = x.flatten(start_dim=1, end_dim=-1)
        x = self.fc(x)
        return x
```

```
if __name__ == '__main__':
    # x = torch.randn([2, 3, 32, 32])
    model = Model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(model)
```

```
Model(
  (model): Sequential(
    (0): Conv2d(3, 5, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (2): Conv2d(5, 10, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (fc): Sequential(
    (0): Linear(in_features=160, out_features=80, bias=True)
    (1): Linear(in_features=80, out_features=10, bias=True)
  )
)
```

```
# from torchsummary import summary
# 通过summary打印输出：
if __name__ == '__main__':
    # x = torch.randn([2, 3, 32, 32])
    model = Model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # print(model)
    summary(model, (3, 32, 32), 2)
```

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1             [2, 5, 16, 16]             140
         MaxPool2d-2               [2, 5, 8, 8]               0
            Conv2d-3              [2, 10, 8, 8]           1,260
         MaxPool2d-4              [2, 10, 4, 4]               0
            Linear-5                    [2, 80]          12,880
            Linear-6                    [2, 10]             810
================================================================
Total params: 15,090
Trainable params: 15,090
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.02
Forward/backward pass size (MB): 0.04
Params size (MB): 0.06
Estimated Total Size (MB): 0.12
----------------------------------------------------------------
```

