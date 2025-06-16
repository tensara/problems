import torch

def solution(X, Y, B: int, D: int):
    l1_norm = torch.sum(torch.abs(X), dim=1, keepdim=True)
    l1_norm = l1_norm + 1e-10
    Y[:] = X / l1_norm