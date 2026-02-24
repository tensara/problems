---
slug: "mxfp8-quantize"
title: "MXFP8 Quantization"
difficulty: "MEDIUM"
author: "sarthak"
tags: ["quantization"]
gpus: ["B200"]
---

Quantize an input FP32 matrix into MXFP8 (Microscaling FP8) using TorchAO's [MXTensor reference path](https://github.com/pytorch/ao/blob/main/torchao/prototype/mx_formats/mx_tensor.py). 

The quantization contract uses:
- Block size of 32 elements along the K dimension.
- Per-block E8M0 scales.
- FP8 E4M3 data bytes.

For more information regarding the MXFP8 format, check out the [MXFP8 specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).

## Input
- $a$: `fp32` pointer to a row-major tensor of shape $M \times K$
- $M$, $K$: dimensions of $a$ (with $K$ divisible by 32)

## Output
- $q$: `uint8` pointer, MXFP8 payload bytes (E4M3 storage bytes) of shape $M \times K$
- $scale$: `uint8` pointer, per-block E8M0 scale bytes in row-major layout of shape $M \times K/32$

## Notes
- The required layout is row-major blocked order (no additional swizzle).
- Verification dequantizes both reference and submitted outputs via TorchAO MXTensor dequantization and checks closeness.
