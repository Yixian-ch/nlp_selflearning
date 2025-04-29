import numpy as np

def matrix_multiplication(m1:np.array, m2:np.array) -> np.array :
# check if the multiplication is legal
    if m1.shape[1] != m2.shape[0]:
        raise ValueError("Matrix' size does not support multiplication")
    new_matrix = np.zeros((m1.shape[0], m2.shape[1]),dtype=int)

    for i in range(m1.shape[0]): # raws
        for j in range(m2.T.shape[0]): # from columns to raws to reduce loop
            new_matrix[i,j] = np.dot(m1[i],m2.T[j])

    return new_matrix 


m1 = np.array(
    [[1,2,3],
    [4,5,6]]
)

m2 = np.array(
    [[2,2],
    [3,3],
    [4,4]]
)

print(matrix_multiplication(m1,m2), m1 @ m2)

import numpy as np
import time

# 创建一个超大的矩阵，比如 5000x5000
size = 5000

# 用 float32 创建矩阵
A32 = np.random.rand(size, size).astype(np.float32)
B32 = np.random.rand(size, size).astype(np.float32)

# 用 float64 创建矩阵
A64 = np.random.rand(size, size).astype(np.float64)
B64 = np.random.rand(size, size).astype(np.float64)

# 测试 float32 相乘用时
start_time = time.time()
C32 = A32 @ B32
time32 = time.time() - start_time
print(f"Float32 multiplication time: {time32:.4f} seconds")

# 测试 float64 相乘用时
start_time = time.time()
C64 = A64 @ B64
time64 = time.time() - start_time
print(f"Float64 multiplication time: {time64:.4f} seconds")
