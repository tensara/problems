---
slug: "tanh"
title: "Tanh"
difficulty: "EASY"
author: "sarthak"
tags: ["cuda-basics", "parallel-computing", "neural-networks"]
parameters:
  - name: "input"
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
    
  - name: "m"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Perform the Tanh activation function on an input matrix:
$$
C[i][j] = \text{tanh}(A[i][j])
$$

The Tanh function is defined as:
$$
\text{tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

## Input:
- Matrix $A$ of size $M \times N$ containing floating-point values

## Output:
- Matrix $C$ of size $M \times N$ containing the Tanh activation values

## Notes:
- Both matrices $\text{A}$ and $\text{C}$ are stored in row-major order
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/22_Tanh.py).