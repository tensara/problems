---
slug: "gemm-relu-divide"
title: "GEMM with ReLU and Divide"
difficulty: "MEDIUM"
author: "codex"
tags: ["kernelbench", "gemm", "activation-function", "exact-port"]
---

Perform a matrix multiplication using learned weights and bias, then apply ReLU and divide by a scalar:
$$
Y = \frac{\mathrm{ReLU}(X W^T + b)}{d}
$$

This is an exact-port-style Tensara adaptation of a KernelBench Level 2 module. The learned
`weight` and `bias` tensors are materialized as deterministic testcase inputs so the runtime
contract remains explicit.

## Input
- `x` of shape `(batch_size, in_features)`
- `weight` of shape `(out_features, in_features)`
- `bias` of shape `(out_features,)`
- `divisor` as a scalar float

## Output
- `output` of shape `(batch_size, out_features)`

## Notes
- `weight` and `bias` correspond to a deterministically initialized `nn.Linear`
- ReLU is applied before the scalar divide
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/63_Gemm_ReLU_Divide.py)
