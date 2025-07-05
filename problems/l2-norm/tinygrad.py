from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(X, Y, B: int, D: int):
    l2_norm = (X * X).sum(axis=1, keepdim=True).sqrt()
    l2_norm = l2_norm + 1e-10
    Y.assign(X / l2_norm)
    Y.realize() 