---
slug: "hard-sigmoid"
title: "Hard Sigmoid"
difficulty: "EASY"
author: "sarthak"
tags: ["activation-function"]
---

Perform the Hard Sigmoid activation function on an input matrix:
$$
C[i][j] = \text{hard\_sigmoid}(A[i][j])
$$

The Hard Sigmoid function is defined as:
$$
\text{hard\_sigmoid}(x) = \begin{cases}
0 & \text{if } x \leq -3 \\
1 & \text{if } x \geq 3 \\
\frac{x + 3}{6} & \text{otherwise}
\end{cases}
$$

## Input:
- Matrix $A$ of size $M \times N$ containing floating-point values

## Output:
- Matrix $C$ of size $M \times N$ containing the Hard Sigmoid activation values

## Notes:
- Both matrices $\text{A}$ and $\text{C}$ are stored in row-major order
- The Hard Sigmoid function is a piecewise linear approximation of the standard Sigmoid function
