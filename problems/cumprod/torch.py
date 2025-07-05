import torch

def solution(input_tensor, output_tensor, n):
    output_tensor[:] = torch.cumprod(input_tensor, dim=0)