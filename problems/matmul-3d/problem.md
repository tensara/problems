---
slug: "matmul-3d"
title: "3D Tensor-Matrix Multiplication"
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

  - name: "n" 
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "m"
    type: "size_t"
    pointer: "false"
    constant: "false"
    
  - name: "k"
    type: "size_t"
    pointer: "false"
    constant: "false"
  
  - name: "l"
    type: "size_t"
    pointer: "false"
    constant: "false"
    
---

Perform 3D tensor-matrix multiplication of two tensors:
$$
C[i][j][l] = \sum_{k=0}^{K-1} A[i][j][k] \cdot B[k][l]
$$

## Input
- Tensor $A$ of size $N \times M \times K$
- Matrix $B$ of size $K \times L$

## Output
- Tensor $C$ of size $N \times M \times L$

## Notes:
- All tensors $\text{A}$, $\text{B}$, and $\text{C}$ are stored in row-major order
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/10_3D_tensor_matrix_multiplication.py)