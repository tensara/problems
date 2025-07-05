from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(anchor, positive, negative, loss, B, E, margin):
    pos_dist = ((anchor - positive) * (anchor - positive)).sum(axis=1).sqrt()
    neg_dist = ((anchor - negative) * (anchor - negative)).sum(axis=1).sqrt()
    losses = (pos_dist - neg_dist + margin).maximum(0)
    loss.assign(losses.mean())
    loss.realize() 