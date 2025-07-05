from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit
import tinygrad.nn as nn

@TinyJit
def solution(X, gamma, beta, Y, B: int, F: int, D1: int, D2: int):
    Y.assign(X.layernorm(axis=(1, 2, 3), eps=1e-5) * gamma + beta)
    Y.realize() 