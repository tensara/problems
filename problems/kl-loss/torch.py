import torch

def solution(predictions, targets, output, n: int):
    eps = 1e-10
    pred_safe = predictions.clamp(min=eps)
    target_safe = targets.clamp(min=eps)
    
    element_wise_kl = target_safe * (torch.log(target_safe) - torch.log(pred_safe))
    
    output[:] = torch.where(targets > 0, element_wise_kl, torch.zeros_like(element_wise_kl))