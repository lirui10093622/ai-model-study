import numpy as np

arr = np.empty((1, 2, 3), dtype=np.float32, order='F')
for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        for k in range(arr.shape[2]):
            arr[i][j][k] = i + j + k

print("arr: ", arr, sep="\n")

arr_like = np.empty_like(arr)
for i in range(arr_like.shape[0]):
    for j in range(arr_like.shape[1]):
        for k in range(arr_like.shape[2]):
            arr_like[i][j][k] = (i + 1) * (j + 1) * (k + 1)

print("arr_like: ", arr_like, sep="\n")

arr_eye = np.eye(5)
print("arr_eye: ", arr_eye, sep="\n")


identity_matrix = np.identity(5)
print("identity_matrix: ", identity_matrix, sep="\n")