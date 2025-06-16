import torch

def solution(A, B, C, n, m, k, l):
    C[:] = torch.einsum('nmk,kl->nml', A, B)