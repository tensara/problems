from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(input, dim, output, shape, ndim):
    output.assign(input.argmax(axis=dim))
    output.realize()