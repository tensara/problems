---
slug: "diagonal-matmul"
title: "Diagonal Matrix Multiplication"
difficulty: "EASY"
author: "sarthak"
tags: ["matmul"]
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