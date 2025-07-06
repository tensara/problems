from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(input_a, input_b, output_c, m, k):
    output_c.assign(input_a @ input_b)
    output_c.realize() 