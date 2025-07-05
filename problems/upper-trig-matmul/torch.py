import torch

def solution(input_a, input_b, output_c, n):
    a_triu = torch.triu(input_a)
    b_triu = torch.triu(input_b)
    output_c[:] = torch.matmul(a_triu, b_triu) 