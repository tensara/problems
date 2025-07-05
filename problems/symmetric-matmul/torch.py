import torch

def solution(input_a, input_b, output_c, n):
    output_c[:] = torch.matmul(input_a, input_b) 