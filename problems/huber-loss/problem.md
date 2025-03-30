---
slug: "huber-loss"
title: "Huber Loss"
difficulty: "EASY"
author: "sarthak"
tags: ["cuda-basics", "loss-functions", "regression", "machine-learning"]
parameters:
  - name: "predictions"
    type: "[VAR]"
    pointer: "true"
    const: "true"
  
  - name: "targets"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "output" 
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "n"
    type: "size_t"
    pointer: "false"
    constant: "false"

---

Compute the element-wise Huber Loss (specifically, Smooth L1 Loss, which is Huber Loss with $\delta=1$) between two input tensors, `predictions` and `targets`.

The Smooth L1 Loss function is defined as:
$$
\text{loss}(x, y) = \frac{1}{n} \sum_{i=1}^n z_i
$$
where $x$ represents predictions, $y$ represents targets, and $z_i$ is given by:
$$
z_i = \begin{cases} 
      0.5 (x_i - y_i)^2 & \text{if } |x_i - y_i| < 1 \\
      |x_i - y_i| - 0.5 & \text{otherwise} 
      \end{cases}
$$

This problem asks you to compute the **element-wise** loss *before* the summation/averaging step. That is, compute $z_i$ for each element $i$.

$$
\text{output}[i] = z_i
$$

## Input:
- Tensor `predictions` of size $N$ 
- Tensor `targets` of size $N$

## Output:
- Tensor `output` of size $N$, where `output[i]` contains the element-wise Huber Loss $z_i$.

## Notes:
- All tensors are flat 1D arrays (or treated as such) and stored contiguously in memory.
- The value $\delta$ for Huber loss is fixed at 1.
