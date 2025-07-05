import torch

def solution(input, output, n, m):
    output[:] = torch.sigmoid(input) 