from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(A, W, b, C, B: int, N: int, M: int):
    C.assign((A @ W.transpose() + b).relu())
    C.realize()
    return C 