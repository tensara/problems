from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(input, kernel_size, stride, padding, dilation, output, H, W):
    input_reshaped = input.reshape(1, 1, H, W)
    result = input_reshaped.max_pool2d(kernel_size=(kernel_size, kernel_size), stride=(stride, stride), padding=(padding, padding), dilation=(dilation, dilation))
    output.assign(result.reshape(result.shape[2], result.shape[3]))
    output.realize()