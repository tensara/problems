from tinygrad.tensor import Tensor
from tinygrad import Device, TinyJit

@TinyJit
def solution(predictions, targets, output, n: int):
    eps = 1e-10
    pred_safe = predictions.clamp(eps, float('inf'))
    target_safe = targets.clamp(eps, float('inf'))
    
    element_wise_kl = target_safe * (target_safe.log() - pred_safe.log())
    output.assign((targets > 0).where(element_wise_kl, element_wise_kl.zeros_like()))
    output.realize()