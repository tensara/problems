from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(input_tensor, output_tensor, n):
    output_tensor.assign(input_tensor.cumprod(axis=0))
    output_tensor.realize() 