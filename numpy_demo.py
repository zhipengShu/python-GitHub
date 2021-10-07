import numpy as np

# import copy


# array = np.array([[1, 2, 3],
#                   [4, 3, 2]])
# print(array.ndim)
# # array.size: Number of elements in the array
# print(array.size)
# print(array.shape)

# array1 = np.array([[1, 2, 3],
#                    [4, 3, 2]], dtype=np.int32)
# array2 = np.array([[1, 2, 3],
#                    [4, 3, 2]], dtype=np.float64)
# print(array1)
# print(array1.dtype)
# print(array2)
# print(array2.dtype)

# array1 = np.zeros((2, 2), dtype=np.int64)
# array2 = np.empty((2, 2), dtype=np.float64)
# array3 = np.ones((2, 2), dtype=np.int32)
# print(array1)
# print(type(array1))
# print(array2)
# print(array3)

# a = np.arange(0, 10, 2)
# print(a)
# print(type(a))
# b = np.linspace(0, 15, 6)
# print(b)
# print(type(b))
# c = b.reshape(2, -1)
# print(c)

# a = np.arange(2, 9, 2).reshape(2, -1)
# b = np.arange(2, 9, 2).reshape(2, -1)
# c = a * b
# d = np.dot(a, b)
# print(a)
# print(b)
# print(c)
# print(d)

# a = np.random.random((2, 3))
# print(a)
# print(np.sum(a))
# print(np.max(a))
# print(np.min(a))

# a = np.random.random((3, 4))
# b = np.argmin(a)
# c = np.argmax(a)
# print(a)
# print(b)
# print(c)

# a = np.arange(2, 14).reshape(3, -1)
# print(a)
# print(a.mean())
# print(a.mean(axis=0))
# print(a.mean(axis=1))
# print(np.median(a))
# print(np.median(a, axis=0))
# print(np.median(a, axis=1))
# print(np.cumsum(a))

# a = np.arange(14, 2, -1).reshape(3, -1)
# print(a)
# print(np.sort(a, axis=None))
# print(np.transpose(a))
# print(a.T)

# a = np.arange(3, 15).reshape(3, 4)
# print(a)
# print(a[1][1])
# print(a[2][3])

# a = np.arange(3, 15).reshape(3, 4)
# print(a)
# print(a.flat)
# print(a.flatten())
# print(list(a.flat))
# for i in a.flatten():
#     if i != 14:
#         print(i, end=" ")
#     else:
#         print(i)
# print('------------------')
# for j in list(a.flat):
#     print(j, end=" ")

# a = np.array([[1, 2, 1]])
# b = np.array([[2, 4, 3]])
# print(np.concatenate((a, b, a, b), axis=0))
# print(np.concatenate((a, b, a, b), axis=1))
# print(np.vstack((a, b)))
# print(a.shape, np.vstack((a, b)).shape)
# print(np.hstack((a, b)))
# print(a.shape, np.hstack((a, b)).shape)

# a = np.array([1, 2, 3, 4]).reshape(2, -1)
# print(a, end='       ')
# print(a.shape, end='\n\n')
#
# print(a[np.newaxis, :], end='       ')
# print(a[np.newaxis, :].shape, end='\n\n')
#
# print(a[:, np.newaxis, :], end='       ')
# print(a[:, np.newaxis, :].shape, end='\n\n')
#
# print(a[:, :, np.newaxis], end='       ')
# print(a[:, :, np.newaxis].shape)

# a = np.arange(12).reshape(3, -1)
# print(a)
# print(np.split(a, 4, axis=1))
# print(np.split(a, 3, axis=0))
# print(np.vsplit(a, 3))
# print(np.hsplit(a, 2))
# print(type(np.vsplit(a, 3)))

# # 直接赋值，只传递对象的引用
# a = 10
# b = a
# print(id(a))
# print(id(b))

# # 浅拷贝 copy（）、[:]
# hello = dict()
# a = [10, 20, [30, 70], hello]
# c = copy.copy(a)
# d = a.copy()
# e = a[:]
#
# a[2].append(88)
# a[3]["world"] = 90
# a[3]["china"] = "good"
# a[1] = 99

# a[0] = 80
# print(id(a[0]))
# e[1] = 80
# print(id(e[1]))

# print(id(a))
# print(id(c))
# print(id(d))
# print(id(e))

# print(id(a[0]))
# print(id(c[0]))
# print(id(d[0]))
# print(id(e[0]))
# print(id(a[1]))
# print(id(c[1]))
# print(id(d[1]))
# print(id(e[1]))

# print(a)
# print(e)
# print(c)
# print(d)

# ee = dict()
# print(id(ee))
# ee[0] = 5
# print(ee)
# print(id(ee))
# ee[10] = 33
# print(ee)
# print(id(ee))

# a = np.array([[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]])
# print(a)
# print(a.shape)
# b = a.transpose((1, 2, 0))
# # b = np.transpose(a, (1, 2, 0))
# print(b)
# print(b.shape)
# print(b.size)

a = np.array([[2, 3, 4, 5],
              [3, 3, 4, 6],
              [2, 5, 6, 9],
              [5, 7, 7, 8]])
# print(a.transpose())
# print(a.transpose((1, 0)))
print(np.transpose(a, (1, 0)))
print(a.T)
