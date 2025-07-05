from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(predictions, targets, output, n):
    output.assign((1 - predictions * targets).maximum(0))
    output.realize() 