from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(input, output, n, m):
    output.assign(input.sigmoid())
    output.realize()