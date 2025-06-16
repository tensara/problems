import torch

def solution(X, Y, B: int, D: int):
    l2_norm = torch.norm(X, p=2, dim=1, keepdim=True)
    l2_norm = l2_norm + 1e-10
    Y[:] = X / l2_norm