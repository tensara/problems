import torch

def solution(X, Y, size):
    norm = torch.norm(X, p='fro')
    Y[:] = X / norm