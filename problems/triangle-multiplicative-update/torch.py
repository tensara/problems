import torch


def solution(left, right, mask, output, B, N, H):
    mask_expanded = mask.unsqueeze(-1)
    left_masked = left * mask_expanded
    right_masked = right * mask_expanded

    left_bh = left_masked.permute(0, 3, 1, 2).reshape(B * H, N, N)
    right_bh_t = right_masked.permute(0, 3, 2, 1).reshape(B * H, N, N)
    out_bh = torch.bmm(left_bh, right_bh_t)
    output[:] = out_bh.reshape(B, H, N, N).permute(0, 2, 3, 1)
