---
slug: "lower-trig-matmul"
title: "Lower Triangular Matrix Multiplication"
difficulty: "MEDIUM"
author: "sarthak" # Assuming the same author, let me know if this should change
tags: ["cuda-basics", "linear-algebra"] # Adjusted tags slightly
parameters:
  - name: "input_a"
    type: "[VAR]"
    pointer: "true"
    const: "true"
  
  - name: "input_b"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "output_c" 
    type: "[VAR]"
    pointer: "true"
    const: "false"

  # Assuming square matrices for lower triangular
  - name: "n"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Perform matrix multiplication of two lower triangular matrices:
$$
C = A \cdot B
$$

Where A and B are lower triangular matrices.

The result C will also be a lower triangular matrix.

## Input
- Lower triangular matrix $A$ of size $N \times N$
- Lower triangular matrix $B$ of size $N \times N$

## Output
- Lower triangular matrix $C$ of size $N \times N$

## Notes:
- All matrices $\text{A}$, $\text{B}$, and $\text{C}$ are stored in row-major order.
- A matrix $L$ is lower triangular if $L_{ij} = 0$ for all $i < j$.
