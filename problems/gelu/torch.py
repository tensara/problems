import torch

def solution(input_tensor, output_tensor, n, m):
    output_tensor[:] = torch.nn.functional.gelu(input_tensor)