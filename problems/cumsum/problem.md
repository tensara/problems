---
slug: "cumsum"
title: "Cumulative Sum"
difficulty: "MEDIUM"
author: "sarthak"
tags: ["cuda-basics", "parallel-computing", "scan", "prefix-sum"]
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

Compute the cumulative sum (also known as prefix sum or scan) of an input array:
$$
\text{output}[i] = \sum_{j=0}^{i} \text{input}[j]
$$

The cumulative sum at each position is the sum of all elements up to and including that position.

## Input:
- Vector $\text{input}$ of size $\text{N}$

## Output:
- Vector $\text{output}$ of size $\text{N}$ containing cumulative sums

## Notes:
- The first element of the output is equal to the first element of the input
