import torch

def solution(input, output, n, m):
    output[:] = torch.nn.functional.softplus(input) 