---
slug: "soft-plus"
title: "Softplus"
difficulty: "EASY"
author: "sarthak"
tags: ["activation-function"]
parameters:
  - name: "input"
    type: "[VAR]"
    pointer: "true"
    const: "true"
  
  - name: "output"
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "n" 
    type: "size_t"
    pointer: "false"
    constant: "false"
    
  - name: "m"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Perform the Softplus activation function on an input matrix:
$$
C[i][j] = \text{softplus}(A[i][j])
$$

The Softplus function is defined as:
$$
\text{softplus}(x) = \ln(1 + e^x)
$$

It can be seen as a smooth approximation of the ReLU function.

## Input:
- Matrix $A$ of size $M \times N$ containing floating-point values

## Output:
- Matrix $C$ of size $M \times N$ containing the Softplus activation values

## Notes:
- Both matrices $\text{A}$ and $\text{C}$ are stored in row-major order
- Softplus is a smooth approximation to the ReLU function and ensures a non-zero gradient for all input values
- Unlike ReLU, which has a sharp transition at x=0, Softplus provides a smooth transition
