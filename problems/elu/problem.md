---
slug: "elu"
title: "ELU"
difficulty: "EASY"
author: "sarthak"
tags: ["activation-function"]
---

Perform the ELU (Exponential Linear Unit) activation function on an input matrix:
$$
C[i][j] = \begin{cases} 
A[i][j] & \text{if } A[i][j] > 0 \\
\alpha \cdot (e^{A[i][j]} - 1) & \text{if } A[i][j] \leq 0 
\end{cases}
$$

The ELU function is defined as:
$$
f(x) = \begin{cases} 
x & \text{if } x > 0 \\
\alpha \cdot (e^x - 1) & \text{if } x \leq 0 
\end{cases}
$$

Where α is a parameter controlling the value to which an ELU saturates for negative inputs.

## Input:
- Matrix $A$ of size $M \times N$ containing floating-point values
- Parameter $\alpha$ (default value: 1.0)

## Output:
- Matrix $C$ of size $M \times N$ containing the ELU activation values

## Notes:
- Both matrices $\text{A}$ and $\text{C}$ are stored in row-major order
- ELU has smoother transitions at x=0 compared to ReLU and helps mitigate the "dying ReLU" problem
