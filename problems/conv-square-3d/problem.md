---
slug: "conv-square-3d"
title: "3D Square Convolution"
difficulty: "HARD" 
author: "sarthak"
tags: ["convolution"]
parameters:
  - name: "A"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "B" 
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "C" 
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "size"
    type: "size_t"
    pointer: "false"
    constant: "false"
    
  - name: "K" 
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Perform 3D convolution between an input volume and a cubic kernel:
$$
\text{C}[i,j,k] = \sum_{x=0}^{K-1}\sum_{y=0}^{K-1}\sum_{z=0}^{K-1} \text{A}[i+x,j+y,k+z] \cdot \text{B}[x,y,z]
$$

The convolution operation slides the 3D kernel over the input volume, computing the sum of element-wise multiplications at each position. Zero padding is used at the boundaries.

## Input:
- Volume $\text{A}$ of size $\text{size} \times \text{size} \times \text{size}$ (input volume)
- Volume $\text{B}$ of size $K \times K \times K$ (cubic convolution kernel)
- $K$ is odd and smaller than $\text{size}$

## Output:
- Volume $\text{C}$ of size $\text{size} \times \text{size} \times \text{size}$ (convolved volume)

## Notes:
- All volumes $\text{A}$, $\text{B}$, and $\text{C}$ are stored in row-major order
- Use zero padding at the boundaries where the kernel extends beyond the input volume
- The kernel is centered at each position, with $(K-1)/2$ elements on each side in all dimensions
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/54_conv_standard_3D__square_input__square_kernel.py)