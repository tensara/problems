import torch

def solution(A, B, C, size: int, K: int):    
    input_reshaped = A.view(1, 1, A.size(0), A.size(1), A.size(2))
    kernel_reshaped = B.view(1, 1, B.size(0), B.size(1), B.size(2))
    
    padding = B.size(0) // 2
    
    result = torch.nn.functional.conv3d(
        input_reshaped,
        kernel_reshaped,
        padding=padding
    )
    
    C[:] = result.view(A.size(0), A.size(1), A.size(2))
