---
slug: "batch-norm"
title: "Batch Normalization"
difficulty: "MEDIUM"
author: "sarthak"
tags: ["normalization"]
parameters:
  - name: "X"
    type: "float"
    pointer: "true"
    const: "true"

  - name: "Y"
    type: "float"
    pointer: "true"
    const: "false"

  - name: "B"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "F"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "D1"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "D2"
    type: "size_t"
    pointer: "false"
    constant: "false"

---

Implement Batch Normalization over the batch dimension (B) for each feature channel in a 4D tensor.

The formula for Batch Normalization is:
$$
\text{y} = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}}
$$
where the mean $\mathrm{E}[x]$ and variance $\mathrm{Var}[x]$ are computed over the batch dimension (B) for each feature channel independently. $\epsilon$ is a small value added to the variance for numerical stability.

## Input:
- Tensor $\text{X}$ of shape $(\text{B}, \text{F}, \text{D1}, \text{D2})$ (input data)
- Epsilon $\epsilon$ (a small float, typically 1e-5)

## Output:
- Tensor $\text{Y}$ of shape $(\text{B}, \text{F}, \text{D1}, \text{D2})$ (normalized data)

## Notes:
- Compute the mean and variance across the batch dimension $\text{B}$ independently for each feature channel $\text{F}$.
- The statistics (mean and variance) are computed independently for each spatial location $(D1, D2)$ in each feature channel.
- Use $\epsilon = 10^{-5}$
- For simplicity, this implementation focuses on the core normalization without learnable parameters (gamma and beta) and without tracking running statistics.
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/33_BatchNorm.py)