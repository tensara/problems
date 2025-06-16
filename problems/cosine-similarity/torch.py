import torch

def solution(predictions, targets, output, n, d):
    output[:] = 1 - torch.nn.functional.cosine_similarity(predictions, targets, dim=-1)