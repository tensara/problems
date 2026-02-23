from tinygrad.tensor import Tensor
from tinygrad import TinyJit

@TinyJit
def solution(input, output, M, N):
    x = input.reshape(M, N)
    result = x.log_softmax(axis=1)
    output.assign(result.reshape(-1))
    output.realize()