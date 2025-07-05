from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(A, B, C, N: int, K: int):
    input_reshaped = A.reshape(1, 1, N)
    kernel_reshaped = B.reshape(1, 1, K)
    padding = K // 2
    result = input_reshaped.conv2d(kernel_reshaped, padding=padding)
    C.assign(result.reshape(N))
    C.realize()