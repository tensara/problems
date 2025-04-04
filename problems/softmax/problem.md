---
slug: "softmax"
title: "Softmax"
difficulty: "MEDIUM" 
author: "sarthak"
tags: ["activation-function", "normalization"]
parameters:
  - name: "input"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "dim"
    type: "int"
    pointer: "false"
    constant: "false"

  - name: "output"
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "shape"
    type: "size_t"
    pointer: "true"
    constant: "true"

  - name: "ndim"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Compute the softmax function over a specified dimension of an input tensor:
$$
\text{softmax}(x_i) = \frac{\exp(x_i)}{\sum_{j=1}^{S_d} \exp(x_j)}
$$

where $x_i$ represents elements along the specified dimension $d$, and $S_d$ is the size of dimension $d$.

## Input:
- Tensor `input` of arbitrary shape $S_1 \times S_2 \times \cdots \times S_n$
- `dim` ($d$): Dimension to compute softmax over (0-based indexing)
- `shape`: Array containing the dimensions of the input tensor
- `ndim` ($n$): Number of dimensions in the input tensor

## Output:
- Tensor `output` with the same shape as input, containing the softmax probabilities

## Notes:
- The input tensor is stored in row-major order
- The output values should be in the range (0, 1)
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/23_Softmax.py)
