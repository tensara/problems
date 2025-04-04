---
slug: "diagonal-matmul"
title: "Diagonal Matrix Multiplication"
difficulty: "EASY"
author: "sarthak"
tags: ["matmul"]
parameters:
  - name: "diagonal_a"
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
    
  - name: "m" 
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Perform matrix multiplication of a diagonal matrix with another matrix:
$$
C[i][j] = A[i] \cdot B[i][j]
$$

## Input
- Diagonal $A$ of size $N$
- Matrix $B$ of size $N \times M$

## Output
- Matrix $C$ of size $N \times M$

## Notes:
- The diagonal matrix is represented by a 1D tensor $A$
- All matrices $\text{B}$ and $\text{C}$ are stored in row-major order
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/12_Matmul_with_diagonal_matrices_.py)