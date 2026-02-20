---
slug: "matmul-4d"
title: "4D Tensor-Matrix Multiplication"
difficulty: "HARD"
author: "sarthak"
tags: ["matmul"]
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