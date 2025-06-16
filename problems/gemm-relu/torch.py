import torch

def solution(A, W, b, C, B: int, N: int, M: int):
    result = torch.matmul(A, W.t()) + b
    C[:] = torch.nn.functional.relu(result)