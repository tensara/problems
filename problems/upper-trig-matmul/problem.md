---
slug: "upper-trig-matmul"
title: "Upper Triangular Matrix Multiplication"
difficulty: "MEDIUM"
author: "sarthak" 
tags: ["matmul"]
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

  # Assuming square matrices for upper triangular
  - name: "n"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Perform matrix multiplication of two upper triangular matrices:
$$
C = A \cdot B
$$

Where A and B are upper triangular matrices.

The result C will also be an upper triangular matrix.

## Input
- Upper triangular matrix $A$ of size $N \times N$
- Upper triangular matrix $B$ of size $N \times N$

## Output
- Upper triangular matrix $C$ of size $N \times N$

## Notes:
- All matrices $\text{A}$, $\text{B}$, and $\text{C}$ are stored in row-major order.
- A matrix $L$ is upper triangular if $L_{ij} = 0$ for all $i > j$.
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/14_Matmul_for_upper_triangular_matrices.py)