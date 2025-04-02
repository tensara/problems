---
slug: "cumprod"
title: "Cumulative Product"
difficulty: "MEDIUM"
author: "sarthak"
tags: ["cuda-basics", "parallel-computing", "scan", "prefix-product"]
parameters:
  - name: "input"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "output" 
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "N"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Compute the cumulative product (also known as prefix product or scan) of an input array:
$$
\text{output}[i] = \prod_{j=0}^{i} \text{input}[j]
$$

The cumulative product at each position is the product of all elements up to and including that position.

## Input:
- Vector $\text{input}$ of size $\text{N}$

## Output:
- Vector $\text{output}$ of size $\text{N}$ containing cumulative products

## Notes:
- The first element of the output is equal to the first element of the input
- Be careful about numerical stability with very large or very small numbers
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/90_cumprod.py)