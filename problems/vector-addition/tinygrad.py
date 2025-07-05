from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(d_input1, d_input2, d_output, n):
    d_output.assign(d_input1 + d_input2)
    d_output.realize() 