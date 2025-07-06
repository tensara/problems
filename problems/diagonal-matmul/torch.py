import torch

def solution(diagonal_a, input_b, output_c, n: int, m: int):
    output_c[:] = torch.diag(diagonal_a) @ input_b