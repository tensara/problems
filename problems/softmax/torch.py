import torch

def solution(input, dim, output, shape, ndim):
    output[:] = torch.nn.functional.softmax(input, dim=dim) 