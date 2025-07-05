from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(input_tensor, output_tensor, n, m, alpha):
    output_tensor.assign(input_tensor.elu(alpha=alpha))
    output_tensor.realize() 