---
slug: "selu"
title: "SELU"
difficulty: "EASY"
author: "sarthak"
tags: ["activation-function"]
---

Perform the SELU (Scaled Exponential Linear Unit) activation function on an input matrix:
$$
C[i][j] = \text{selu}(A[i][j])
$$

The SELU function is defined as:
$$
\text{selu}(x) = \lambda \cdot  (\max (0, x) + \min (0, \alpha \cdot (e^x - 1)))
$$

Where $\lambda \approx 1.0507$ and $\alpha \approx 1.67326$ are predetermined constants.

## Input:
- Matrix $A$ of size $M \times N$ containing floating-point values

## Output:
- Matrix $C$ of size $M \times N$ containing the SELU activation values

## Notes:
- Both matrices $\text{A}$ and $\text{C}$ are stored in row-major order
- SELU was introduced in the paper "Self-Normalizing Neural Networks" to help preserve the mean and variance of signals flowing through neural networks
