---
slug: "layer-norm"
title: "Layer Normalization"
difficulty: "MEDIUM"
author: "sarthak"
tags: ["normalization"]
---

Implement Layer Normalization over the last 3 dimensions (F, D1, D2) of a 4D tensor.

The formula for Layer Normalization is:
$$
\text{y} = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
$$
where the mean $\mathrm{E}[x]$ and variance $\mathrm{Var}[x]$ are computed over the normalization dimensions (F, D1, D2) for each element in the first dimension (B). $\gamma$ and $\beta$ are learnable affine parameters (elementwise scale and shift), and $\epsilon$ is a small value added to the variance for numerical stability.

## Input:
- Tensor $\text{X}$ of shape $(\text{B}, \text{F}, \text{D1}, \text{D2})$ (input data)
- Vector $\text{gamma}$ of shape $(\text{F}, \text{D1}, \text{D2})$ (scale parameters)
- Vector $\text{beta}$ of shape $(\text{F}, \text{D1}, \text{D2})$ (shift parameters)
- Epsilon $\epsilon$ (a small float, typically 1e-5)

## Output:
- Tensor $\text{Y}$ of shape $(\text{B}, \text{F}, \text{D1}, \text{D2})$ (normalized data)

## Notes:
- Compute the mean and variance across the last three dimensions $(\text{F}, \text{D1}, \text{D2})$ independently for each sample in the batch $\text{B}$.
- Apply the normalization using the computed mean/variance and the provided $\gamma$ and $\beta$.
- Use $\epsilon = 10^{-5}$
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/40_LayerNorm.py)