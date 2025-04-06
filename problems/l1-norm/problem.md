---
slug: "l1-norm"
title: "L1 Normalization"
difficulty: "EASY"
author: "sarthak"
tags: ["normalization"]
parameters:
  - name: "X"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "Y"
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "B"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "D"
    type: "size_t"
    pointer: "false"
    constant: "false"

---

Implement L1 Normalization for a 2D tensor. L1 normalization is a technique where each row of the input tensor is normalized by the sum of the absolute values of its elements.

The formula for L1 Normalization is:
$$
\text{y} = \frac{x}{\sum |x_i|}
$$
where the sum of absolute values is computed across the second dimension (D) for each element in the first dimension (B).

## Input:
- Tensor $\text{X}$ of shape $(\text{B}, \text{D})$ (input data)

## Output:
- Tensor $\text{Y}$ of shape $(\text{B}, \text{D})$ (normalized data)

## Notes:
- For numerical stability, you may need to add a small epsilon $\epsilon = 10^{-10}$ to the denominator to avoid division by zero.
- After normalization, the L1 norm of each row should be approximately 1.0.
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/38_L1Norm_.py)