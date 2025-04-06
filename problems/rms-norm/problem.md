---
slug: "rms-norm"
title: "RMS Normalization"
difficulty: "EASY"
author: "sarthak"
tags: ["normalization"]
parameters:
  - name: "X"
    type: "[VAR]"
    pointer: "true"
    const: "true"
  
  - name: "Y"
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "B"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "N"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Implement RMS (Root Mean Square) Normalization for a 2D tensor.

Normalize the input by dividing each element by the root mean square of the features in each sample. More formally, compute:

$$
\text{y} = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}}
$$

where the mean is computed along the feature dimension for each sample in the batch independently. $\epsilon$ is a small value added to the denominator for numerical stability.

## Input:
- Tensor $\text{X}$ of shape $(\text{B}, \text{N})$ is the input data where $\text{B}$ = batch size and $\text{N}$ = number of features

## Output:
- Tensor $\text{Y}$ with the same shape as input (normalized data)

## Notes:
- For each sample, the RMS is calculated over the feature dimension (dimension 1).
- Use $\epsilon = 10^{-5}$ for numerical stability.
