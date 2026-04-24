import torch

def solution(input_tensor, mask, p, output, n, m):
    scale = 1.0 / (1.0 - p) if p < 1.0 else 0.0
    output[:] = (input_tensor * mask) * scale
