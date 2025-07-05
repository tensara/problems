from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(input_tensor, output_tensor, n, m):
    output_tensor.assign(input_tensor.hardsigmoid())
    output_tensor.realize() 