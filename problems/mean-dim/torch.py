import torch

def solution(input, dim, output, shape, ndim):
    output[:] = torch.mean(input, dim=dim, keepdim=True) 