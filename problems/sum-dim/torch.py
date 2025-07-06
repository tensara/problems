import torch

def solution(input, dim, output, shape, ndim):
    output[:] = torch.sum(input, dim=dim, keepdim=True) 