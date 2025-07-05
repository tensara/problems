from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(input, kernel_size: int, stride: int, padding: int, output, H: int, W: int):
    input_reshaped = input.reshape(1, 1, H, W)
    result = input_reshaped.avg_pool2d(kernel_size=(kernel_size, kernel_size), stride=(stride, stride), padding=(padding, padding))
    output.assign(result.reshape(result.shape[2], result.shape[3]))
    output.realize()