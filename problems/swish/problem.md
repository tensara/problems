---
slug: "swish"
title: "Swish"
difficulty: "EASY"
author: "sarthak"
tags: ["activation-function"]
---

Perform the Swish activation function on an input matrix:
$$
C[i][j] = A[i][j] \cdot \sigma(A[i][j])
$$

The Swish function is defined as:
$$
\text{swish}(x) = x \cdot \sigma(x)
$$

## Input:
- Matrix $A$ of size $M \times N$ containing floating-point values

## Output:
- Matrix $C$ of size $M \times N$ containing the Swish activation values

## Notes:
- Both matrices $\text{A}$ and $\text{C}$ are stored in row-major order
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/25_Swish.py)