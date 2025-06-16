import torch

def solution(X, gamma, beta, Y, B: int, F: int, D1: int, D2: int):

    Y[:] = torch.nn.functional.layer_norm(
        X, 
        (F, D1, D2), 
        weight=gamma, 
        bias=beta, 
        eps=1e-5
    )
