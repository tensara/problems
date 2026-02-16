---
slug: "hinge-loss"
title: "Hinge Loss"
difficulty: "EASY"
author: "sarthak"
tags: ["loss-function"]
parameters:
  - name: "predictions"
    type: "float"
    pointer: "true"
    const: "true"
  
  - name: "targets"
    type: "float"
    pointer: "true"
    const: "true"

  - name: "output" 
    type: "float"
    pointer: "true"
    const: "false"

  - name: "n"
    type: "size_t"
    pointer: "false"
    constant: "false"

---

Compute the Hinge Loss between predictions and binary targets (-1 or 1).

The Hinge Loss function is defined as:
$$
\text{loss}(x, y) = \max(0, 1 - xy)
$$
where $x$ represents predictions and $y$ represents binary targets in {-1, 1}.

For a batch of inputs, the loss is averaged:
$$
\text{Loss} = \frac{1}{n} \sum_{i=1}^n \max(0, 1 - x_i y_i)
$$

This implementation focuses on computing the element-wise loss before averaging:
$$
\text{output}[i] = \max(0, 1 - x_i y_i)
$$

## Input:
- Tensor `predictions` of size $N$ (real-valued predictions)
- Tensor `targets` of size $N$ (binary values in {-1, 1})

## Output:
- Tensor `output` of size $N$ containing element-wise hinge loss values

## Notes:
- All tensors are flat 1D arrays (or treated as such) and stored contiguously in memory
- Target values must be either -1 or 1
- The loss is non-negative and equals 0 only when the prediction has the correct sign and magnitude $\geq$ 1
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/100_HingeLoss.py)