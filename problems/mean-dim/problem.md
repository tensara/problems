---
slug: "mean-dim"
title: "Mean Over Dimension"
difficulty: "EASY" 
author: "sarthak"
tags: ["reduction"]
parameters:
  - name: "input"
    type: "float"
    pointer: "true"
    const: "true"

  - name: "dim"
    type: "int"
    pointer: "false"
    constant: "false"

  - name: "output"
    type: "float"
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

Perform mean reduction over a specified dimension of an input tensor:
$$
\text{output}[i_1,\ldots,i_{d-1},1,i_{d+1},\ldots,i_n] = \frac{1}{S_d}\sum_{i_d=0}^{S_d-1} \text{input}[i_1,\ldots,i_d,\ldots,i_n]
$$

where $d$ is the dimension to reduce over, $n$ is the number of dimensions, and $S_d$ is the size of dimension $d$.

## Input:
- Tensor `input` of arbitrary shape $S_1 \times S_2 \times \cdots \times S_n$
- `dim` ($d$): Dimension to reduce over (0-based indexing)
- `shape`: Array containing the dimensions of the input tensor
- `ndim` ($n$): Number of dimensions in the input tensor

## Output:
- Tensor `output` with shape $S_1 \times \cdots \times S_{d-1} \times 1 \times S_{d+1} \times \cdots \times S_n$
  - The reduced dimension is kept with size 1 (keepdim=True)

## Notes:
- The input tensor is stored in row-major order
- The reduction should maintain numerical stability by using appropriate accumulation techniques
- The output tensor preserves the dimensionality of the input tensor with the reduced dimension having size 1
- This problem computes the arithmetic mean (average) of elements along the specified dimension
