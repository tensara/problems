---
slug: "matrix-scalar"
title: "Matrix Scalar Multiplication"
difficulty: "EASY"
author: "sarthak"
tags: ["matmul", "scalar"]
parameters:
  - name: "input_matrix"
    type: "[VAR]"
    pointer: "true"
    const: "true"
  
  - name: "scalar"
    type: "[VAR]"
    pointer: "false"
    const: "true"

  - name: "output_matrix" 
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "n" 
    type: "size_t"
    pointer: "false"
    constant: "false"
    
---

Perform multiplication of a matrix with a scalar value:
$$
C[i][j] = A[i][j] \cdot s
$$
where $s$ is the scalar value.

## Input:
- Matrix $A$ of size $\text{n} \times \text{n}$
- Scalar value $s$

## Output:
- Matrix $C = s \cdot A$ of size $\text{n} \times \text{n}$

## Notes:
- Matrix $\text{A}$ is stored in row-major order
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/5_Matrix_scalar_multiplication.py)