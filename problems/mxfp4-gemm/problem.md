---
slug: "mxfp4-gemm"
title: "MXFP4 GEMM"
difficulty: "HARD"
author: "sarthak"
tags: ["quantization", "mxfp4", "matrix-multiplication"]
gpus: ["B200"]
---

Compute matrix multiplication where both operands are stored in MXFP4 format. The equation below defines reference semantics for correctness.

$$
c_{ij} = \sum_{\ell=0}^{K-1} A_{\mathrm{dequant},i\ell} B_{\mathrm{dequant},j\ell}.
$$

## Input
- $q_a$: packed MXFP4 payload bytes for matrix $A$ of logical shape $M \times K$ (as a `uint8_t` pointer)
- $scale_a$: per-block E8M0 scale bytes for $A$ (as a `uint8_t` pointer)
- $q_b$: packed MXFP4 payload bytes for matrix $B$ of logical shape $N \times K$ (as a `uint8_t` pointer)
- $scale_b$: per-block E8M0 scale bytes for $B$ (as a `uint8_t` pointer)
- $M$, $N$, $K$: matrix dimensions ($K$ divisible by 32)

## Output
- $c$: FP32 matrix of shape $M \times N$, with $c = A_{\mathrm{dequant}}B_{\mathrm{dequant}}^T$

## Notes
- The reference implementation in this problem calls [torch.nn.functional.scaled_mm](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_mm.html) and does **not** materialize $A_{\mathrm{dequant}}$ or $B_{\mathrm{dequant}}$.
