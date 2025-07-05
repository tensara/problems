import torch

def solution(A, B, C, N: int, K: int):
    input_reshaped = A.view(1, 1, -1)
    kernel_reshaped = B.view(1, 1, -1)
    
    padding = K // 2
    
    C[:] = torch.nn.functional.conv1d(
        input_reshaped,
        kernel_reshaped,
        padding=padding
    ).view(-1)