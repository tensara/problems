---
slug: "mse_loss"
title: "Mean Squared Error Loss"
difficulty: "EASY" 
author: "harmya"
tags: ["loss-function"]
parameters:
  - name: "predictions"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "targets"
    type: "[VAR]"
    pointer: "true"
    const: "true"

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

Compute the Mean Squared Error (MSE) loss between predictions and targets:
$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

where $y_i$ represents the predictions, $\hat{y}_i$ represents the targets, and $N$ is the total number of elements.

## Input:
- Tensor `predictions` of arbitrary shape $S_1 \times S_2 \times \cdots \times S_n$
- Tensor `targets` of the same shape as predictions
- `shape`: Array containing the dimensions of the input tensors
- `ndim` ($n$): Number of dimensions in the input tensors

## Output:
- Scalar `output` containing the mean squared error loss

## Notes:
- The input tensors are stored in row-major order
- The MSE loss is a scalar value that represents the average squared difference between predictions and targets
- This implementation should handle tensors of arbitrary shapes
- For numerical stability, consider intermediate steps carefully