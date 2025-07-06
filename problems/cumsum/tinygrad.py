from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(input_tensor, output_tensor, n):
    output_tensor.assign(input_tensor.cumsum(axis=0).float())
    output_tensor.realize()