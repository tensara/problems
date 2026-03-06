---
slug: "mxfp4-gemv"
title: "MXFP4 GEMV"
difficulty: "HARD"
author: "sarthak"
tags: ["quantization", "mxfp4", "matmul", "vector"]
gpus: ["B200"]
---

Compute matrix-vector multiplication where both matrix $A$ and vector $x$ are stored in MXFP4 format.

$$
y_i = \sum_{\ell=0}^{K-1} A_{\mathrm{dequant},i\ell} \, x_{\mathrm{dequant},\ell}.
$$

Equivalently, $y = A_{\mathrm{dequant}} x_{\mathrm{dequant}}$ where $A_{\mathrm{dequant}} \in \mathbb{R}^{M \times K}$ and $x_{\mathrm{dequant}} \in \mathbb{R}^{K}$.

## Input
- $q_a$: MXFP4 payload bytes for matrix $A$ of shape $M \times K$ (row-major)
- $scale_a$: per-block E8M0 scale bytes for $A$, logical shape $M \times K/32$
- $q_x$: MXFP4 payload bytes for vector $x$, represented as logical shape $1 \times K$
- $scale_x$: per-block E8M0 scale bytes for $x$, logical shape $1 \times K/32$
- $M$, $K$: dimensions ($K$ divisible by 32)

## Output
- $y$: FP32 vector of shape $M$

## Notes
- The reference dequantizes MXFP4 inputs with TorchAO MXTensor semantics and performs FP32 `matmul`.
- Scale tensors in this problem are row-major blocked order (not swizzled).
- Correctness is based on dequantized semantics, not bitwise equality of quantized payloads.
