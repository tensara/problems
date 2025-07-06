import torch

def solution(input_matrix, weight_matrix, bias, scaling_factor, output, batch_size, in_features, out_features):
    z = torch.matmul(input_matrix, weight_matrix.t()) + bias
    output[:] = scaling_factor * z * torch.sigmoid(z) 

