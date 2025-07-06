from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(predictions, targets, output, n, d):
    dot_product = (predictions * targets).sum(axis=-1)

    pred_norm = (predictions * predictions).sum(axis=-1).sqrt()
    target_norm = (targets * targets).sum(axis=-1).sqrt()
    
    cos_sim = dot_product / (pred_norm * target_norm)
    
    output.assign(1 - cos_sim)
    output.realize()