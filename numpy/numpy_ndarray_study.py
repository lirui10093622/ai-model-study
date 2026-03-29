import numpy as np


def print_ndarray(nd):
    print(nd)
    print(f'flags: {nd.flags}, shape: {nd.shape}, strides: {nd.strides}, ndim: {nd.ndim}, size: {nd.size}, itemsize: {nd.itemsize}, dtype: {nd.dtype}')

nd = np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]], [[7, 7, 7], [8, 8, 8], [9, 9, 9]]])
print_ndarray(nd)
