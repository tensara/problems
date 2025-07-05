---
title: "Final Vector Addition"
slug: "final-vector-addition"
difficulty: "EASY"
author: "Somesh Kar"
parameters:
  - name: a
    type: torch.Tensor
  - name: b
    type: torch.Tensor
---

## Problem Statement
Perform element-wise addition of two vectors:
$$
c_i = a_i + b_i
$$

## Input
- Tensors `a` and `b` of length `N`

## Output
- Tensor `c` of length `N` containing the element-wise sum