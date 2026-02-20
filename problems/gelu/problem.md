---
slug: "gelu"
title: "GELU"
difficulty: "EASY"
author: "harmya"
tags: ["activation-function"]
---

Perform the GELU (Gaussian Error Linear Unit) activation function on an input matrix:
$$
C[i][j] = \text{GELU}(A[i][j])
$$

The GELU function is defined as:
$$
\text{GELU}(x) = x \cdot \Phi(x)
$$

where $\Phi(x)$ is the cumulative distribution function of the standard normal distribution. 

A common approximation for GELU is:
$$
\text{GELU}(x) \approx 0.5x \cdot (1 + \tanh(\sqrt{2/\pi} \cdot (x + 0.044715x^3)))
$$

## Input:
- Matrix $A$ of size $M \times N$ containing floating-point values

## Output:
- Matrix $C$ of size $M \times N$ containing the GELU activation values

## Notes:
- Both matrices $\text{A}$ and $\text{C}$ are stored in row-major order
- You should implement the approximation formula for GELU defined above
- GELU is commonly used in modern transformer-based neural networks like BERT and GPT