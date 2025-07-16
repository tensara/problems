---
slug: "gemm-multiply-leakyrelu"
title: "GEMM + Element-wise Multiply + LeakyReLU Fusion"
difficulty: "HARD"
author: "sarthak"
tags: ["gemm", "multiply", "leakyrelu", "fusion"]
parameters:
  - name: "matrix_a"
    type: "[VAR]"
    pointer: "true"
    const: "true"
  
  - name: "matrix_b"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "matrix_c"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "alpha"
    type: "float"
    pointer: "false"
    constant: "false"

  - name: "output_matrix" 
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "m"
    type: "size_t"
    pointer: "false"
    constant: "false"
    
  - name: "n" 
    type: "size_t"
    pointer: "false"
    constant: "false"
    
  - name: "k"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Perform fused GEMM (General Matrix Multiplication) followed by element-wise multiplication followed by LeakyReLU activation:

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
- Slope parameter $\alpha$ for LeakyReLU

## Output
- Matrix $O$ of size $M \times N$

## Notes:
- All matrices $A$, $B$, $C$, and $O$ are stored in row-major order
- LeakyReLU allows small negative values to pass through, preventing dying neurons
- The fusion of these operations can significantly reduce memory bandwidth requirements
- Consider optimizing for different values of $\alpha$ (typically small positive values like 0.01) 