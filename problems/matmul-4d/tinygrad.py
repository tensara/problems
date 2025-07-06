from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(A, B, C, b, i, j, l, k):
    output = A.reshape(b*i*j, l) @ B
    C.assign(output.reshape(b, i, j, k))
    C.realize() 