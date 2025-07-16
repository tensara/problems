---
slug: "matmul-swish-scaling"
title: "Matrix Multiplication + Swish + Scaling Fusion"
difficulty: "HARD"
author: "sarthak"
tags: ["matmul", "swish", "scaling", "fusion"]
parameters:
  - name: "input_a"
    type: "[VAR]"
    pointer: "true"
    const: "true"
  
  - name: "input_b"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "output_matrix" 
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "scale"
    type: "float"
    pointer: "false"
    constant: "false"

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

Perform fused matrix multiplication followed by Swish activation followed by scaling:

$$
O[i][j] = \text{scale} \times \text{swish}\left(\sum_{k=0}^{K-1} A[i][k] \cdot B[k][j]\right)
$$

where $\text{swish}(x) = x \cdot \sigma(x)$ and $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function.

This operation consists of three steps:
1. Matrix multiplication: $C[i][j] = \sum_{k=0}^{K-1} A[i][k] \cdot B[k][j]$
2. Swish activation: $S[i][j] = C[i][j] \cdot \sigma(C[i][j])$
3. Scaling: $O[i][j] = \text{scale} \times S[i][j]$

## Input
- Matrix $A$ of size $M \times K$
- Matrix $B$ of size $K \times N$
- Scaling factor $\text{scale}$

## Output
- Matrix $O$ of size $M \times N$

## Notes:
- All matrices $A$, $B$, and $O$ are stored in row-major order
- Swish is a smooth, non-monotonic activation function that often outperforms ReLU
- The fusion of these operations can provide significant performance benefits by reducing memory bandwidth
- Consider optimizing for numerical stability when computing the sigmoid function 