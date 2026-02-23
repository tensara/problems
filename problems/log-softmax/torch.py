import torch

def solution(input, output, M, N):
    output[:] = torch.nn.functional.log_softmax(input.view(M, N), dim=1).view(-1)