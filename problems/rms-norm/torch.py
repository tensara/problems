import torch

def solution(X, Y, B, N):
    rms = torch.sqrt(torch.mean(X ** 2, dim=1, keepdim=True) + 1e-5)
    Y[:] = X / rms 