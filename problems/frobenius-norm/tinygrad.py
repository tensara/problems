from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(X, Y, size):
    norm = ((X * X).sum()).sqrt()
    Y.assign(X / norm)
    Y.realize() 