import torch

def solution(rgb_image, grayscale_output, height: int, width: int, channels: int):
    rgb_flat = rgb_image.reshape(-1, 3)
    weights = torch.tensor([[0.299], [0.587], [0.114]], device=rgb_image.device, dtype=rgb_image.dtype)
    
    gray_flat = torch.matmul(rgb_flat, weights)
    grayscale_output[:] = gray_flat.reshape(height, width)