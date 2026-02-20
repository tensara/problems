---
slug: "lower-trig-matmul"
title: "Lower Triangular Matrix Multiplication"
difficulty: "MEDIUM"
author: "sarthak" 
tags: ["matmul"]
---

Perform matrix multiplication of two lower triangular matrices:
$$
C = A \cdot B
$$

Where A and B are lower triangular matrices.

The result C will also be a lower triangular matrix.

## Input
- Lower triangular matrix $A$ of size $N \times N$
- Lower triangular matrix $B$ of size $N \times N$

## Output
- Lower triangular matrix $C$ of size $N \times N$

## Notes:
- All matrices $\text{A}$, $\text{B}$, and $\text{C}$ are stored in row-major order.
- A matrix $L$ is lower triangular if $L_{ij} = 0$ for all $i < j$.
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/15_Matmul_for_lower_triangular_matrices.py)