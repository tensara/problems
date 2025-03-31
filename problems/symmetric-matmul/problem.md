---
slug: "symmetric-matmul"
title: "Symmetric Matrix Multiplication"
difficulty: "MEDIUM"
author: "sarthak"
tags: ["cuda-basics", "parallel-computing"]
parameters:
  - name: "input_a"
    type: "[VAR]"
    pointer: "true"
    const: "true"
  
  - name: "input_b"
    type: "[VAR]"
    pointer: "true"
    const: "true"
  
  - name: "output_c"
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "n" 
    type: "size_t"
    pointer: "false"
    constant: "false"
    
---

Perform multiplication of two symmetric matrices:
$$
C[i][j] = \sum_{k=0}^{N-1} A[i][k] \cdot B[k][j]
$$

## Input
- $A$ is a symmetric matrix of size $N \times N$
- $B$ is a symmetric matrix of size $N \times N$ 

## Output
- Matrix $C = AB$ of size $N \times N$

## Notes:
- All matrices $\text{A}$, $\text{B}$, and $\text{C}$ are stored in row-major order
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/13_Matmul_for_symmetric_matrices.py).