---
slug: "nvfp4-gemv"
title: "NVFP4 GEMV"
difficulty: "HARD"
author: "sarthak"
tags: ["quantization", "nvfp4", "matmul", "vector"]
gpus: ["B200"]
---

Compute matrix-vector multiplication where both matrix $A$ and vector $x$ are stored in NVFP4 format. The equation below defines conceptual dequantization semantics for correctness:

$$
y_i = \sum_{\ell=0}^{K-1} A_{\mathrm{dequant},i\ell} \, x_{\mathrm{dequant},\ell}.
$$

This is equivalent to $y = A_{\mathrm{dequant}} x_{\mathrm{dequant}}$ with:
- $A_{\mathrm{dequant}} \in \mathbb{R}^{M \times K}$
- $x_{\mathrm{dequant}} \in \mathbb{R}^{K}$
- $y \in \mathbb{R}^{M}$

## Input
- $q_a$: packed NVFP4 E2M1 payload bytes for matrix $A$ of logical shape $M \times K$
- $scale_a$: NVFP4 per-block FP8 scale bytes for $A$, logical shape $M \times K/16$
- $sf\_g\_a$: global NVFP4 encode factor for $A$
- $q_x$: packed NVFP4 E2M1 payload bytes for vector $x$, represented as logical shape $1 \times K$
- $scale_x$: NVFP4 per-block FP8 scale bytes for $x$, logical shape $1 \times K/16$
- $sf\_g\_x$: global NVFP4 encode factor for $x$
- $M$, $K$: dimensions ($K$ divisible by 16)

## Output
- $y$: FP16 vector of shape $M$

## Notes
- The reference implementation dequantizes NVFP4 inputs with [FlashInfer decode](https://docs.flashinfer.ai/generated/flashinfer.fp4_quantization.e2m1_and_ufp8sf_scale_to_float.html) semantics, then computes GEMV as `matmul` in FP32 before casting to FP16 output.
- $scale\_a$ and $scale\_x$ are already in NVFP4 swizzled scale layout expected by the decode path, do **not** apply an additional swizzle.
