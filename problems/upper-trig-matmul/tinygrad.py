from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(input_a, input_b, output_c, n):
    a_triu = input_a.triu()
    b_triu = input_b.triu()
    output_c.assign(a_triu @ b_triu)
    output_c.realize() 