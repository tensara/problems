import torch

def solution(A, B, C, b, i, j, l, k):
    C[:] = torch.einsum("bijl,lk->bijk", A, B) 