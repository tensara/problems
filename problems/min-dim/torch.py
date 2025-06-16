import torch

def solution(input, dim, output, shape, ndim):
    output[:] = torch.min(input, dim=dim, keepdim=True)[0] 