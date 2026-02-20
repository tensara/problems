---
slug: "conv-1d"
title: "1D Convolution"
difficulty: "EASY"
author: "sarthak"
tags: ["convolution"]
---

Perform 1D convolution between an input signal and a kernel with zero padding and centered kernel.

Let $r = \frac{K-1}{2}$. Out-of-bounds accesses to $\text{A}$ are treated as zero.

$$
\text{C}[i] = \sum_{j=0}^{K-1} \text{A}[\,i + j - r\,] \cdot \text{B}[j]
$$

The convolution operation slides the kernel over the input signal, computing the sum of element-wise multiplications at each position. Zero padding is used at the boundaries.

## Input:

- Vector $\text{A}$ of size $\text{N}$ (input signal)
- Vector $\text{B}$ of size $\text{K}$ (convolution kernel)

## Output:

- Vector $\text{C}$ of size $\text{N}$ (convolved signal)

## Notes:

- $\text{K}$ is odd and smaller than $\text{N}$
- Use zero padding at the boundaries where the kernel extends beyond the input signal
- The kernel is centered at each position, with $(K-1)/2$ elements on each side
- Output size is $N$ (same as input) due to padding
- This matches PyTorch `torch.nn.functional.conv1d(..., padding=K//2)` (cross-correlation, kernel is not flipped)
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/67_conv_standard_1D.py)
