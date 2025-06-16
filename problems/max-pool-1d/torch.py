import torch

def solution(input, kernel_size, stride, padding, dilation, output, H):
    # Reshape input for torch.nn.functional.max_pool1d
    input_reshaped = input.view(1, 1, H)
    
    # Apply max pooling
    result = torch.nn.functional.max_pool1d(
        input_reshaped,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation
    )
    
    # Reshape output back to 1D and assign
    output[:] = result.view(-1) 