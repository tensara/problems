import torch

def solution(X, Y, B: int, F: int, D1: int, D2: int):
    bn = torch.nn.BatchNorm2d(
        num_features=F,
        affine=False,
        track_running_stats=False,
        eps=1e-5
    )
    Y[:] = bn(X)