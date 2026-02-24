---
slug: "mxfp8-dequantize"
title: "MXFP8 Dequantization"
difficulty: "EASY"
author: "sarthak"
tags: ["quantization", "mxfp8"]
gpus: ["B200"]
---

Dequantize an MXFP8-encoded matrix back to FP32. See the [MXFP8 format](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) for more background.

## Input
- $q$: MXFP8 payload bytes for matrix $A$ of shape $M \times K$ (given as a `uint8_t` pointer)
- $scale$: per-block E8M0 scale bytes for $A$ (given as a `uint8_t` pointer)
- $M$, $K$: matrix dimensions ($K$ divisible by 32)

## Output
- $out$: FP32 matrix of shape $M \times K$

## Notes
- Dequantization semantics match [TorchAO MXTensor](https://github.com/pytorch/ao/blob/main/torchao/prototype/mx_formats/mx_tensor.py) (`to_dtype`) for MXFP8.
