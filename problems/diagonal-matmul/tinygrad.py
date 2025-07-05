from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(diagonal_a, input_b, output_c, n: int, m: int):
    output_c.assign((Tensor.eye(n) * diagonal_a) @ input_b)
    output_c.realize() 