import torch

def solution(predictions, targets, output, n: int):
    output[:] = torch.nn.functional.smooth_l1_loss(predictions, targets, reduction='none', beta=1.0)