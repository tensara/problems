---
slug: "mxfp4-dequantize"
title: "MXFP4 Dequantization"
difficulty: "EASY"
author: "sarthak"
tags: ["quantization", "mxfp4"]
gpus: ["B200"]
---

Dequantize an MXFP4-encoded matrix back to FP32. See the [MXFP4 format](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) for more background.

## Input
- $q$: packed MXFP4 payload bytes for matrix $A$ of shape $M \times K$ (given as a `uint8_t` pointer)
- $scale$: per-block E8M0 scale bytes for $A$ (given as a `uint8_t` pointer)
- $M$, $K$: matrix dimensions ($K$ divisible by 32)

## Output
- $out$: FP32 matrix of shape $M \times K$

## Notes
- Dequantization semantics match [TorchAO MXTensor](https://github.com/pytorch/ao/blob/main/torchao/prototype/mx_formats/mx_tensor.py) (`to_dtype`) for MXFP4.
