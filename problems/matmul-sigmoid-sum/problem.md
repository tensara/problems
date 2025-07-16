---
slug: "matmul-sigmoid-sum"
title: "Matrix Multiplication + Sigmoid + Sum"
difficulty: "HARD"
author: "sarthak"
tags: ["matmul", "reduction", "fusion"]
parameters:
  - name: "input_a"
    type: "[VAR]"
    pointer: "true"
    const: "true"
  
  - name: "input_b"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "output_result" 
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

Perform fused matrix multiplication followed by sigmoid activation followed by summation:

$$
\text{result} = \sum_{i=0}^{M-1} \sum_{j=0}^{N-1} \sigma\left(\sum_{k=0}^{K-1} A[i][k] \cdot B[k][j]\right)
$$

where $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function.

This operation consists of three steps:
1. Matrix multiplication: $C[i][j] = \sum_{k=0}^{K-1} A[i][k] \cdot B[k][j]$
2. Sigmoid activation: $S[i][j] = \sigma(C[i][j])$
3. Sum reduction: $\text{result} = \sum_{i,j} S[i][j]$

## Input
- Matrix $A$ of size $M \times K$
- Matrix $B$ of size $K \times N$

## Output
- Scalar value representing the sum of $\sigma(AB)$

## Notes:
- All matrices $A$ and $B$ are stored in row-major order
