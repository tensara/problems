from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(input_a, input_b, output_c, n):
    output_c.assign(input_a @ input_b)
    output_c.realize() 