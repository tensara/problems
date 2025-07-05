from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(predictions, targets, output, shape, ndim):
    output.assign(((predictions - targets) * (predictions - targets)).mean())
    output.realize() 