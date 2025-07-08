---
slug: "matrix-power"
title: "Matrix Nth Power"
difficulty: "MEDIUM"
author: "nnarek"
tags: ["matmul"]
parameters:
  - name: "input_matrix"
    type: "[VAR]"
    pointer: "true"
    const: "true"
  
  - name: "n"
    type: "size_t"
    pointer: "false"
    const: "true"

  - name: "output_matrix" 
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "size" 
    type: "size_t"
    pointer: "false"
    constant: "false"
    
---

Compute the nth power of a square matrix:
$$
C = A^n
$$
where $A$ is a square matrix and $n$ is a non-negative integer.

## Input:
- Matrix $A$ of size $\text{size} \times \text{size}$
- Integer power $n$ (non-negative integer)

## Output:
- Matrix $C = A^n$ of size $\text{size} \times \text{size}$

## Mathematical Definition:
For a square matrix $A$ and non-negative integer $n$:
- $A^0 = I$
- $A^1 = A$
- $A^2 = A \times A$
- $A^3 = A \times A \times A$
- $A^n = \underbrace{A \times A \times \cdots \times A}_{n \text{ times}}$

## Notes:
- Matrix $A$ is stored in row-major order