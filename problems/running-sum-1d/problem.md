---
slug: "running-sum-1d"
title: "1D Running Sum"
difficulty: "EASY" 
author: "nnarek"
tags: ["convolution"]
parameters:
  - name: "input"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "W" 
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "output" 
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "N"
    type: "size_t"
    pointer: "false"
    constant: "false"
    

    
---

Calculate 1D running sum with fix sized sliding window:
$$
\text{output}[i] = \sum_{j=0}^{W-1} \text{input}[i + j]
$$

The running sum operation slides the window over the input data and computing the sum for each window. Zero padding is used at the boundaries.

## Input:
- Vector $\text{input}$ of size $\text{N}$ (input data)

## Output:
- Vector $\text{output}$ of size $\text{N}$ (output sums)

## Notes:
- $\text{W}$ is odd and smaller than $\text{N}$
- Use zero padding at the boundaries where the window extends beyond the input data
- The window is centered at each position, with $(W-1)/2$ elements on each side