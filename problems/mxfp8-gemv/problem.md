---
slug: "mxfp8-gemv"
title: "MXFP8 GEMV"
difficulty: "MEDIUM"
author: "sarthak"
tags: ["quantization", "mxfp8", "vector"]
gpus: ["B200"]
---

Compute matrix-vector multiplication where both operands are stored in MXFP8 format. The equation below defines reference semantics for correctness.

$$
y_i = \sum_{j=0}^{K-1} A_{\mathrm{dequant},ij} x_{\mathrm{dequant},j}.
$$

arc## Input
- $q_a$: MXFP8 payload bytes for matrix $A$ of shape $M \times K$
- $scale_a$: per-block E8M0 scale bytes for $A$ (logical shape $M \times K/32$)
- $q_x$: MXFP8 payload bytes for vector $x$ of shape $K \times 1$
- $scale_x$: per-block E8M0 scale bytes for $x$ 
- $M$, $K$: matrix dimensions ($K$ divisible by 32)

## Output
- $y$: FP32 vector of shape $M$

## Notes
- The reference dequantizes $q_a$/$\mathit{scale}_a$ and $q_x$/$\mathit{scale}_x$ to FP32 (same MXFP8 dequantization as in the [mxfp8-quantize](/problems/mxfp8-quantize) / [mxfp8-dequantize](/problems/mxfp8-dequantize) problems), then computes $y = A_{\mathrm{dequant}}\,x_{\mathrm{dequant}}$ in FP32. However, optimized kernels should perform dequantization on-the-fly and avoid materializing full FP32 $A_{\mathrm{dequant}}$ or $x_{\mathrm{dequant}}$.
