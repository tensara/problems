import torch

def solution(input, dim, output, shape, ndim):
    output[:] = torch.max(input, dim=dim, keepdim=True)[0] 