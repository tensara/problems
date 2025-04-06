---
slug: "rms-norm"
title: "RMS Normalization"
difficulty: "MEDIUM"
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

  - name: "total_size"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "dims_size"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Implement RMS (Root Mean Square) Normalization for a tensor of arbitrary shape.

RMS Normalization is a technique often used in transformers and other deep learning models, particularly in language models. It normalizes the input by dividing each element by the root mean square of the features in each sample.

The formula for RMS Normalization is:

$$
\text{y} = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}}
$$

where the mean is computed along the feature dimension for each sample in the batch independently. $\epsilon$ is a small value added to the denominator for numerical stability.

## Input:
- Tensor $\text{X}$ of shape $(\text{B}, \text{N}, *)$ (input data)

## Output:
- Tensor $\text{Y}$ with the same shape as input (normalized data)

## Notes:
- For each sample, the RMS is calculated over the feature dimension (dimension 1).
- Use $\epsilon = 10^{-5}$ for numerical stability.
