import torch

def solution(input_a, input_b, output_c, n):
    a_tril = torch.tril(input_a)
    b_tril = torch.tril(input_b)

    output_c[:] = torch.matmul(a_tril, b_tril) 