import torch

def solution(input, kernel_size, stride, padding, dilation, output, H, W):
    # Reshape input for torch.nn.functional.max_pool2d
    input_reshaped = input.view(1, 1, H, W)
    
    # Apply max pooling
    result = torch.nn.functional.max_pool2d(
        input_reshaped,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation
    )
    
    # Reshape output back to 2D and assign
    output[:] = result.view(result.size(2), result.size(3)) 