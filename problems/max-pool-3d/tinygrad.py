from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(input, kernel_size, stride, padding, dilation, output, H, W, D):
    x = input.reshape(1, 1, D, H, W)
    result = x.max_pool2d(kernel_size=(kernel_size, kernel_size, kernel_size), 
                         stride=(stride, stride, stride),
                         padding=(padding, padding, padding),
                         dilation=(dilation, dilation, dilation))
    result = result.reshape(result.shape[2], result.shape[3], result.shape[4])
    output[:] = result.realize()