---
slug: "nvfp4-dequantize"
title: "NVFP4 Dequantization"
difficulty: "MEDIUM"
author: "sarthak"
tags: ["quantization", "nvfp4"]
gpus: ["B200"]
---

Dequantize an NVFP4-encoded matrix back to FP32. See the [NVFP4 format blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) for more background.

## Input
- $q$: packed NVFP4 E2M1 payload bytes for matrix $A$ of logical shape $M \times K$ (given as a `uint8_t` pointer)
- $scale$: NVFP4 per-block FP8 scale bytes (given as a `uint8_t` pointer)
- $sf_g$: global NVFP4 encode factor
- $M$, $K$: matrix dimensions ($K$ divisible by 16)

## Output
- $out$: FP32 matrix of shape $M \times K$

## Notes
- Use [FlashInfer's NVFP4 dequantization]((https://docs.flashinfer.ai/generated/flashinfer.fp4_quantization.e2m1_and_ufp8sf_scale_to_float.html) ) semantics for correctness; submissions are compared against this function in FP32 space.
- The $scale$ input uses the same swizzled 128x4 layout as in [nvfp4-quantize](/problems/nvfp4-quantize) (see the [cuBLAS 1D Block Scaling Factors Layout](https://docs.nvidia.com/cuda/cublas/#d-block-scaling-factors-layout)).