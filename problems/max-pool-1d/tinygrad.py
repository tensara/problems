from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(input, kernel_size, stride, padding, dilation, output, H):
    x = input.reshape(1, 1, 1, H)
    result = x.max_pool2d(kernel_size=(1, kernel_size),
                         stride=(1, stride),
                         padding=(0, padding),
                         dilation=(1, dilation))
    
    result = result.reshape(-1)
    
    output[:] = result.realize()