from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(A, B, C, size: int, K: int):
    input_reshaped = A.reshape(1, 1, size, size, size)
    kernel_reshaped = B.reshape(1, 1, K, K, K)
    padding = K // 2
    result = input_reshaped.conv2d(kernel_reshaped, padding=(padding, padding, padding))
    C.assign(result.reshape(size, size, size))
    C.realize() 