from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(input_matrix, scalar, output_matrix, n):
    output_matrix.assign(input_matrix * scalar)
    output_matrix.realize() 