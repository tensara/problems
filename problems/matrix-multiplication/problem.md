---
slug: "matrix-multiplication"
title: "Matrix Multiplication"
difficulty: "MEDIUM"
author: "sarthak"
tags: ["matmul"]
---

Perform matrix multiplication of two matrices:
$$
C[i][j] = \sum_{k=0}^{K-1} A[i][k] \cdot B[k][j]
$$

## Input
- Matrix $A$ of size $M \times K$
- Matrix $B$ of size $K \times N$

## Output
- Matrix $C$ of size $M \times N$

## Notes:
- All matrices $\text{A}$, $\text{B}$, and $\text{C}$ are stored in row-major order