import torch

def solution(input, dim, output, shape, ndim):
    output[:] = torch.prod(input, dim=dim, keepdim=True) 