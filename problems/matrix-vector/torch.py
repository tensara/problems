import torch

def solution(input_a, input_b, output_c, m, k):
    output_c[:] = torch.matmul(input_a, input_b) 