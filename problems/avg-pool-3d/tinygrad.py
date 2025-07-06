from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(input, kernel_size: int, stride: int, padding: int, output, H: int, W: int, D: int):
    x = input.reshape(1, 1, D, H, W)
    result = x.avg_pool2d(kernel_size=(kernel_size, kernel_size, kernel_size), 
                         stride=(stride, stride, stride),
                         padding=(padding, padding, padding))
    result = result.reshape(result.shape[2], result.shape[3], result.shape[4])
    output[:] = result.realize()