import torch

def solution(predictions, targets, output, n):
    output[:] = torch.clamp(1 - predictions * targets, min=0)