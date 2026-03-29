import numpy as np

ndarray_empty = np.empty((1, 2, 3), dtype=np.float32, order='F')
for i in range(ndarray_empty.shape[0]):
    for j in range(ndarray_empty.shape[1]):
        for k in range(ndarray_empty.shape[2]):
            ndarray_empty[i][j][k] = i + j + k

print("ndarray_empty: ", ndarray_empty, sep="\n")


ndarray_like = np.empty_like(ndarray_empty)
for i in range(ndarray_like.shape[0]):
    for j in range(ndarray_like.shape[1]):
        for k in range(ndarray_like.shape[2]):
            ndarray_like[i][j][k] = (i + 1) * (j + 1) * (k + 1)
print("ndarray_like: ", ndarray_like, sep="\n")


ndarray_eye = np.eye(5, 7, 1)
print("ndarray_eye: ", ndarray_eye, sep="\n")


ndarray_identity = np.identity(5)
print("ndarray_identity: ", ndarray_identity, sep="\n")

ndarray_ones = np.ones((2, 4))
print("ndarray_ones: ", ndarray_ones, sep="\n")

ndarray_ones_like = np.ones_like(ndarray_empty)
print("ndarray_ones_like: ", ndarray_ones_like, sep="\n")

ndarray_zeros = np.zeros((2, 4))
print("ndarray_zeros: ", ndarray_zeros, sep="\n")

ndarray_zeros_like = np.zeros_like(ndarray_empty)
print("ndarray_zeros_like: ", ndarray_zeros_like, sep="\n")

ndarray_full = np.full((2, 4), 2)
print("ndarray_full: ", ndarray_full, sep="\n")

ndarray_full_array = np.full((2, 3), [1, 2, 3])
print("ndarray_full_array: ", ndarray_full_array, sep="\n")

ndarray_full_like = np.full_like(ndarray_full_array, 2)
print("ndarray_full_like: ", ndarray_full_like, sep="\n")

ndarray_array = np.array([1, 2, 3, 4, 5])
print("ndarray_array: ", ndarray_array, sep="\n")
print("ndarray_array.max(): ", ndarray_array.max(), sep="\n")

array = [1, 2, 3, 4, 5]
print("array: ", array, sep="\n")

ndarray_asarray = np.asarray((("张三", 20, '男'), ("李四", 21, '女')))
print("ndarray_asarray(from tuple): ", ndarray_asarray, sep="\n")

ndarray_asarray = np.asarray([[3, 2, 1], [4, 5, 6]])
print("ndarray_asarray(from array): ", ndarray_asarray, sep="\n")

ndarray_asanyarray = np.asanyarray(array)
print("ndarray_asanyarray: ", ndarray_asanyarray, sep="\n")

ndarray_ascontiguousarray = np.ascontiguousarray(array)
print("ndarray_ascontiguousarray: ", ndarray_ascontiguousarray, sep="\n")

matrix = [[[1], [2], [3]], [[4], [5], [6]]]
print("matrix: ", matrix[0][0][0])

# ndarray_asmatrix = np.asmatrix(matrix)
# print("ndarray_asmatrix: ", ndarray_asmatrix, sep="\n")

matrix = [[1, 2, 3], [4, 5, 6]]
ndarray_asmatrix = np.asmatrix(matrix)
print("ndarray_asmatrix: ", ndarray_asmatrix, sep="\n")

ndarray_astype = ndarray_asmatrix.astype(np.float32)
print("ndarray_astype: ", ndarray_astype, sep="\n")

ndarray_astype = np.astype(ndarray_astype, np.int32)
print("ndarray_astype: ", ndarray_astype, sep="\n")

ndarray_copy = np.copy(ndarray_astype)
print("ndarray_copy: ", ndarray_copy, sep="\n")

ndarray_frombuffer = np.frombuffer(b'hello world', dtype='S2', count=3, offset=1)
print("ndarray_frombuffer: ", ndarray_frombuffer, sep="\n")