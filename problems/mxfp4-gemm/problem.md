---
slug: "mxfp4-gemm"
title: "MXFP4 GEMM"
difficulty: "HARD"
author: "sarthak"
tags: ["quantization", "mxfp4", "matrix-multiplication"]
gpus: ["B200"]
---

Compute matrix multiplication where both matrix $A$ and matrix $B$ are stored in MXFP4 format. The equation below defines reference semantics for correctness; optimized kernels should decode/apply scales on-the-fly and avoid materializing full FP32 $A_{\mathrm{dequant}}$ or $B_{\mathrm{dequant}}$.

$$
c_{ij} = \sum_{\ell=0}^{K-1} A_{\mathrm{dequant},i\ell} \, B_{\mathrm{dequant},j\ell}.
$$

Note: $B$ is stored in row-major as $N \times K$ (i.e. $B_{\mathrm{dequant}}$ is $N \times K$), so the multiplication is effectively $C = A_{\mathrm{dequant}} \, B_{\mathrm{dequant}}^T$.

## Input
- $q_a$: MXFP4 payload bytes for matrix $A$ of shape $M \times K$ (row-major)
- $scale_a$: per-block E8M0 scale bytes for $A$
- $q_b$: MXFP4 payload bytes for matrix $B$ of shape $N \times K$ (row-major; transposed before multiply)
- $scale_b$: per-block E8M0 scale bytes for $B$
- $M$, $N$, $K$: matrix dimensions ($K$ and $N$ divisible by 32)

## Output
- $c$: FP32 matrix of shape $M \times N$ where $c = A_{\mathrm{dequant}} \, B_{\mathrm{dequant}}^T$

## Notes
- We use [torch.scaled_mm](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_mm.html) as the reference implementation for correctness. `scaled_mm` expects the second matrix in column-major layout; the reference therefore passes $B_{\mathrm{dequant}}^T$ (shape $K \times N$) as the second argument so the result logically remains $C = A_{\mathrm{dequant}} \, B_{\mathrm{dequant}}^T$.
- Scale tensors passed as $scale\_a$ / $scale\_b$ are assumed to already be laid out in the same [swizzled blockwise format](https://github.com/pytorch/pytorch/blob/b9698289591834e133d705e6c5c7840e18fb54b8/torch/csrc/Module.cpp#L2722-L2728) that `scaled_mm` uses for MXFP formats. 
- You should treat these pointers as already-swizzled $32 \times 4 \times 4$ layout scale storage and must _not_ apply an additional swizzle.
