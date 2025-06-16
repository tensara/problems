import torch

def solution(input, kernel_size, stride, padding, output, H):
    input_reshaped = input.view(1, 1, H)
    result = torch.nn.functional.avg_pool1d(
        input_reshaped,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )
    output[:] = result.view(result.size(2))