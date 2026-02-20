---
slug: "gemm-multiply-leakyrelu"
title: "GEMM with Element-wise Multiply and LeakyReLU"
difficulty: "MEDIUM"
author: "sarthak"
tags: ["matmul", "activation-function", "fused"]
---

Perform a GEMM (General Matrix Multiplication) followed by element-wise multiplication followed by LeakyReLU activation:

$$
O[i][j] = \text{LeakyReLU}\left(\left(\sum_{k=0}^{K-1} A[i][k] \cdot B[k][j]\right) \cdot C[i][j], \alpha\right)
$$

where $\text{LeakyReLU}(x, \alpha) = \begin{cases} x & \text{if } x \geq 0 \\ \alpha \cdot x & \text{if } x < 0 \end{cases}$

This operation consists of three steps:
1. GEMM operation: $G[i][j] = \sum_{k=0}^{K-1} A[i][k] \cdot B[k][j]$
2. Element-wise multiplication: $M[i][j] = G[i][j] \cdot C[i][j]$
3. LeakyReLU activation: $O[i][j] = \text{LeakyReLU}(M[i][j], \alpha)$

## Input
- Matrix $A$ of size $M \times K$
- Matrix $B$ of size $K \times N$  
- Matrix $C$ of size $M \times N$ (for element-wise multiplication)
- $\alpha$ for LeakyReLU

## Output
- Matrix $O$ of size $M \times N$

## Notes:
- All matrices $A$, $B$, $C$, and $O$ are stored in row-major order