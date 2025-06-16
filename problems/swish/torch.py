import torch

def solution(input, output, n, m):
    output[:] = input * torch.sigmoid(input) 