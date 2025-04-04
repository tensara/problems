---
slug: "matmul-4d"
title: "4D Tensor-Matrix Multiplication"
difficulty: "HARD"
author: "sarthak"
tags: ["matmul"]
parameters:
  - name: "A"
    type: "[VAR]"
    pointer: "true"
    const: "true"
  
  - name: "B"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "C" 
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "b" 
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "i"
    type: "size_t"
    pointer: "false"
    constant: "false"
    
  - name: "j"
    type: "size_t"
    pointer: "false"
    constant: "false"
  
  - name: "l"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "k"
    type: "size_t"
    pointer: "false"
    constant: "false"
    
---

Perform 4D tensor-matrix multiplication of two tensors:
$$
C[b][i][j][k] = \sum_{l=0}^{L-1} A[b][i][j][l] \cdot B[l][k]
$$

## Input
- Tensor $A$ of size $B \times I \times J \times L$
- Matrix $B$ of size $L \times K$

## Output
- Tensor $C$ of size $B \times I \times J \times K$

## Notes:
- All tensors $\text{A}$, $\text{B}$, and $\text{C}$ are stored in row-major order
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/11_4D_tensor_matrix_multiplication.py)