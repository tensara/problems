from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(input_matrix, alpha, output_matrix, rows, cols):
    output_matrix.assign(input_matrix.maximum(input_matrix * alpha))
    output_matrix.realize()