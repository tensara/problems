from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(A, B, C, n, m, k, l):
    output = A.reshape(n*m, k) @ B
    C.assign(output.reshape(n, m, l))
    C.realize() 