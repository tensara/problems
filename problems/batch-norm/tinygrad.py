from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit
import tinygrad.nn as nn

@TinyJit
def solution(X, Y, B: int, F: int, D1: int, D2: int):
    bn = nn.BatchNorm2d(F, eps=1e-5, affine=False, track_running_stats=False)
    Y.assign(bn(X))
    Y.realize() 