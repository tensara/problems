---
slug: "leaky-relu"
title: "Leaky ReLU"
difficulty: "EASY"
author: "sarthak"
tags: ["activation-function"]
parameters:
  - name: "input"
    type: "[VAR]"
    pointer: "true"
    const: "true"
  
  - name: "alpha"
    type: "[VAR]"
    pointer: "false"
    constant: "false"

  - name: "output" 
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "n"
    type: "size_t"
    pointer: "false"
    constant: "false"
    
  - name: "m" 
    type: "size_t"
    pointer: "false"
    constant: "false"

---

Perform the Leaky ReLU (Leaky Rectified Linear Unit) activation function on an input matrix:
$$
\text{C}[i][j] = \max(\alpha \cdot \text{A}[i][j], \text{A}[i][j])
$$
where $\alpha$ is a small positive constant (e.g. 0.01)

The Leaky ReLU function is defined as:
$$
f(x) = \begin{cases} 
x & \text{if } x > 0 \\
\alpha x & \text{if } x \leq 0 
\end{cases}
$$

## Input:
- Matrix $\text{A}$ of size $M \times N$ 
- $\alpha$ value (slope for negative values)

## Output:
- Matrix $\text{C}$ of size $M \times N$

## Notes:
- Both matrices $\text{A}$ and $\text{C}$ are stored in row-major order
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/20_LeakyReLU.py)