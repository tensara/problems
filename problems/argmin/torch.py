import torch

def solution(input, dim, output, shape, ndim):
    output[:] = torch.argmin(input, dim=dim) 