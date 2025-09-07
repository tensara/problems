---
slug: "matmul-swish-scaling"
title: "Matrix Multiplication with Swish and Scaling"
difficulty: "MEDIUM"
author: "sarthak"
tags: ["matmul", "activation-function", "fused"]
parameters:
  - name: "A"
    type: "[VAR]"
    pointer: "true"
    const: "true"
  
  - name: "B"
    type: "[VAR]"
    pointer: "true"
    const: "true"
  
  - name: "scale"
    type: "float"
    pointer: "false"
    constant: "false"

  - name: "output" 
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "M"
    type: "size_t"
    pointer: "false"
    constant: "false"
    
  - name: "N" 
    type: "size_t"
    pointer: "false"
    constant: "false"
    
  - name: "K"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Perform a matrix multiplication followed by Swish activation followed by scaling:

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