import torch

def solution(input_tensor, output_tensor, n, m, alpha):
    output_tensor[:] = torch.nn.functional.elu(input_tensor, alpha=alpha)