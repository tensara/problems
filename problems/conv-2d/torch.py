import torch

def solution(A, B, C, H: int, W: int, Kh: int, Kw: int):
    input_reshaped = A.view(1, 1, H, W)
    kernel_reshaped = B.view(1, 1, Kh, Kw)
    
    padding_h = Kh // 2
    padding_w = Kw // 2
    
    C[:] = torch.nn.functional.conv2d(
        input_reshaped,
        kernel_reshaped,
        padding=(padding_h, padding_w)
    ).view(H, W)