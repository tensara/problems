from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(X, Y, B, N):
    rms = ((X * X).mean(axis=1, keepdim=True) + 1e-5).sqrt()
    Y.assign(X / rms)
    Y.realize() 