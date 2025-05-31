import torch

def solution(input_matrix, alpha, output_matrix, rows, cols):
    output_matrix[:] = torch.nn.functional.leaky_relu(input_matrix, alpha)