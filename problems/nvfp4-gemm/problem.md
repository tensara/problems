---
slug: "nvfp4-gemm"
title: "NVFP4 GEMM"
difficulty: "HARD"
author: "sarthak"
tags: ["quantization", "nvfp4", "matrix-multiplication"]
gpus: ["B200"]
---

Compute matrix multiplication where both matrix $A$ and matrix $B$ are stored in NVFP4 format. The equation below defines conceptual dequantization semantics for correctness:
$$
c_{ij} = \sum_{\ell=0}^{K-1} A_{\mathrm{dequant},i\ell} \, B_{\mathrm{dequant},j\ell}.
$$

Note: $B$ is stored in row-major as $N \times K$ (i.e. $B_{\mathrm{dequant}}$ is $N \times K$), so the multiplication is effectively
$$
C = A_{\mathrm{dequant}} \, B_{\mathrm{dequant}}^T.
$$

## Input
- $q_a$: packed NVFP4 E2M1 payload bytes for matrix $A$ of logical shape $M \times K$
- $scale_a$: NVFP4 per-block FP8 scale bytes for $A$
- $q_b$: packed NVFP4 E2M1 payload bytes for matrix $B$ of logical shape $N \times K$
- $scale_b$: NVFP4 per-block FP8 scale bytes for $B$
- $M$, $N$, $K$: matrix dimensions ($K$ divisible by 16)
- $sf\_g\_a$: global NVFP4 scale factor for $A$
- $sf\_g\_b$: global NVFP4 encode factor for $B$

## Output
- $c$: FP16 matrix of shape $M \times N$, with $c = A_{\mathrm{dequant}}B_{\mathrm{dequant}}^T$

## Notes

- The reference implementation in this problem calls [torch.nn.functional.scaled_mm](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_mm.html) and does **not** materialize $A_{\mathrm{dequant}}$ or $B_{\mathrm{dequant}}$.
