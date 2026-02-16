---
slug: "square-matmul"
title: "Square Matrix Multiplication"
difficulty: "MEDIUM"
author: "sarthak"
tags: ["matmul"]
parameters:
  - name: "input_a"
    type: "float"
    pointer: "true"
    const: "true"
  
  - name: "input_b"
    type: "float"
    pointer: "true"
    const: "true"
  
  - name: "output_c"
    type: "float"
    pointer: "true"
    const: "false"

  - name: "n" 
    type: "size_t"
    pointer: "false"
    constant: "false"
    
---

Perform multiplication of two square matrices:
$$
C[i][j] = \sum_{k=0}^{N-1} A[i][k] \cdot B[k][j]
$$

## Input
- Matrix $A$ of size $N \times N$
- Matrix $B$ of size $N \times N$ 

## Output
- Matrix $C = AB$ of size $N \times N$

## Notes:
- All matrices $\text{A}$, $\text{B}$, and $\text{C}$ are stored in row-major order
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/1_Square_matrix_multiplication_.py)