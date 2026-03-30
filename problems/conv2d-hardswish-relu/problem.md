---
slug: "conv2d-hardswish-relu"
title: "Conv2d with HardSwish and ReLU"
difficulty: "MEDIUM"
author: "codex"
tags: ["kernelbench", "convolution", "activation-function", "exact-port"]
---

Perform a learned 2D convolution, apply HardSwish, and then apply ReLU:
$$
Y = \mathrm{ReLU}(\mathrm{HardSwish}(\mathrm{Conv2d}(X, W, b)))
$$

This is an exact-port-style Tensara adaptation of a KernelBench Level 2 module. The learned
convolution weights and bias are materialized as deterministic testcase inputs.

## Input
- `x` of shape `(batch_size, in_channels, height, width)`
- `weight` of shape `(out_channels, in_channels, kernel_size, kernel_size)`
- `bias` of shape `(out_channels,)`

## Output
- `output` of shape `(batch_size, out_channels, height - kernel_size + 1, width - kernel_size + 1)`

## Notes
- Convolution uses stride `1`, padding `0`, dilation `1`, and groups `1`
- The activation order matters: HardSwish first, ReLU second
- This problem is distinct from the existing normalized `conv2d-relu-hardswish` problem
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/69_Conv2d_HardSwish_ReLU.py)
