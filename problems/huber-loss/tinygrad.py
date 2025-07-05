from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(predictions, targets, output, n: int):
    beta = 1.0
    diff = predictions - targets
    abs_diff = diff.abs()
    quadratic = 0.5 * (diff * diff)
    linear = beta * (abs_diff - 0.5 * beta)
    output.assign((abs_diff <= beta) * quadratic + (abs_diff > beta) * linear)
    output.realize()