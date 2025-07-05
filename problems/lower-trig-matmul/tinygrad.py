from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(input_a, input_b, output_c, n):
    a_tril = input_a.tril()
    b_tril = input_b.tril()
    output_c.assign(a_tril @ b_tril)
    output_c.realize() 