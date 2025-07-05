from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(A, B, C, H: int, W: int, Kh: int, Kw: int):
    input_reshaped = A.reshape(1, 1, H, W)
    kernel_reshaped = B.reshape(1, 1, Kh, Kw)
    padding_h = Kh // 2
    padding_w = Kw // 2
    result = input_reshaped.conv2d(kernel_reshaped, padding=(padding_h, padding_w))
    C.assign(result.reshape(H, W))
    C.realize() 