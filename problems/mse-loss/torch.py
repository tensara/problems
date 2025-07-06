import torch

def solution(predictions, targets, output, shape, ndim):
    output[0] = torch.mean((predictions - targets) ** 2)