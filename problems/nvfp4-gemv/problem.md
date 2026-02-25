---
slug: "nvfp4-gemv"
title: "NVFP4 GEMV"
difficulty: "HARD"
author: "sarthak"
tags: ["quantization", "gemv", "low-precision"]
gpus: ["B200"]
---

Compute matrix-vector multiplication where both operands are stored in NVFP4 format. The equation below defines reference semantics for correctness.

$$
y_i = \sum_{j=0}^{K-1} A_{\mathrm{dequant},ij} x_{\mathrm{dequant},j}.
$$

## Input
- $q_a$: packed NVFP4 E2M1 payload bytes for matrix $A$ of logical shape $M \times K$
- $scale_a$: NVFP4 per-block FP8 scale bytes for $A$
- $q_x$: packed NVFP4 E2M1 payload bytes for row vector $x$ of shape $1 \times K$
- $scale_x$: NVFP4 per-block FP8 scale bytes for $x$
- $M$, $K$: matrix dimensions ($K$ divisible by 16)
- $sf_g_a$: global NVFP4 scale factor for $A$
- $sf_g_x$: global NVFP4 scale factor for $x$

## Output
- $y$: FP32 vector of shape $M$

## Notes
- NVFP4 semantics are preserved per operand (`sf_g` + local block scales).
- The reference executes GEMV as GEMM($K \times 1$) via [`torch.nn.functional.scaled_mm`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_mm.html) when available.
- Fallback is mathematically equivalent dequantize-then-GEMV.
