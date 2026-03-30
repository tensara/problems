---
slug: "matmul-mish-mish"
title: "Matmul with Mish and Mish"
difficulty: "MEDIUM"
author: "codex"
tags: ["kernelbench", "matmul", "activation-function", "exact-port"]
---

Perform a learned linear transform and apply Mish twice:
$$
Y = \mathrm{Mish}(\mathrm{Mish}(X W^T + b))
$$

This is an exact-port-style Tensara adaptation of a KernelBench Level 2 module. The learned
`weight` and `bias` tensors are materialized as deterministic testcase inputs.

## Input
- `x` of shape `(batch_size, in_features)`
- `weight` of shape `(out_features, in_features)`
- `bias` of shape `(out_features,)`

## Output
- `output` of shape `(batch_size, out_features)`

## Notes
- `weight` and `bias` correspond to a deterministically initialized `nn.Linear`
- Mish is applied twice in sequence
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/29_Matmul_Mish_Mish.py)
