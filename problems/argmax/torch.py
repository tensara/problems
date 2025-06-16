import torch

def solution(input, dim, output, shape, ndim):
    output[:] = torch.argmax(input, dim=dim) 