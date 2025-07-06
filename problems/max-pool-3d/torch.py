import torch

def solution(input, kernel_size, stride, padding, dilation, output, H, W, D):
    # Reshape input for torch.nn.functional.max_pool3d
    input_reshaped = input.view(1, 1, H, W, D)
    
    # Apply max pooling
    result = torch.nn.functional.max_pool3d(
        input_reshaped,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation
    )
    
    # Reshape output back to 3D and assign
    output[:] = result.view(result.size(2), result.size(3), result.size(4)) 