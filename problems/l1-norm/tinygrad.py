from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(X, Y, B: int, D: int):
    l1_norm = X.abs().sum(axis=1, keepdim=True)
    l1_norm = l1_norm + 1e-10
    Y.assign(X / l1_norm)
    Y.realize() 