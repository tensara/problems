import torch

def solution(input, kernel_size: int, stride: int, padding: int, output, H: int, W: int, D: int):
    input_reshaped = input.view(1, 1, H, W, D)
    result = torch.nn.functional.avg_pool3d(
        input_reshaped,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )
    output[:] = result.view(result.size(2), result.size(3), result.size(4))