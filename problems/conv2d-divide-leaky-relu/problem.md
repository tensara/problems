---
slug: "conv2d-divide-leaky-relu"
title: "Conv2d with Divide and Leaky ReLU"
difficulty: "MEDIUM"
author: "codex"
tags: ["kernelbench", "convolution", "activation-function", "exact-port"]
---

Perform a learned 2D convolution, divide the result by a scalar, and apply Leaky ReLU:
$$
Y = \mathrm{LeakyReLU}\left(\frac{\mathrm{Conv2d}(X, W, b)}{d}, \alpha\right)
$$

This is an exact-port-style Tensara adaptation of a KernelBench Level 2 module. The learned
convolution weights and bias are materialized as deterministic testcase inputs.

## Input
- `x` of shape `(batch_size, in_channels, height, width)`
- `weight` of shape `(out_channels, in_channels, kernel_size, kernel_size)`
- `bias` of shape `(out_channels,)`
- `divisor` as a scalar float
- `negative_slope` as a scalar float

## Output
- `output` of shape `(batch_size, out_channels, height - kernel_size + 1, width - kernel_size + 1)`

## Notes
- Convolution uses stride `1`, padding `0`, dilation `1`, and groups `1`
- The negative slope is fixed to `0.01` in the source task
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/71_Conv2d_Divide_LeakyReLU.py)
