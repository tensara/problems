---
slug: "dropout"
title: "Dropout"
difficulty: "EASY"
author: "prashantpandeygit"
tags: ["regularization", "neural-networks"]
---

Perform the Dropout regularization operation on an input matrix using a given binary mask:
$$
C[i][j] = \frac{A[i][j] \times \text{mask}[i][j]}{1 - p}
$$

The Dropout operation is defined as:
$$
f(x, m) = \begin{cases} 
\frac{x}{1 - p} & \text{if } m = 1 \\
0 & \text{if } m = 0 
\end{cases}
$$

Where $p$ is the dropout probability and $m$ is the corresponding pregenerated binary mask value.

## Input:
- Matrix $A$ of size $M \times N$ containing floating-point values
- Matrix $\text{mask}$ of size $M \times N$ containing binary values (0.0 or 1.0)
- Parameter $p$ (dropout probability, $0 \leq p < 1$)

## Output:
- Matrix $C$ of size $M \times N$ containing the dropout-applied values

## Notes:
- All matrices are stored in row-major order
- The inverted dropout scaling $\frac{1}{1-p}$ preserves the expected value of the activations
