import torch

def solution(anchor, positive, negative, loss, B, E, margin):
    loss[0] = torch.nn.functional.triplet_margin_loss(
        anchor, positive, negative,
        margin=margin,
        reduction='mean'
    ) 