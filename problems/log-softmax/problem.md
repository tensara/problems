---
slug: "log-softmax"
title: "Log Softmax"
difficulty: "EASY"
author: "tensara"
tags: ["activation-function"]
parameters:
  - name: "input"
    type: "float"
    pointer: true
    const: true
  - name: "output"
    type: "float"
    pointer: true
    const: false
  - name: "M"
    type: "size_t"
    pointer: false
    const: false
  - name: "N"
    type: "size_t"
    pointer: false
    const: false
---

Perform the Log-Softmax activation function on an input matrix row-wise:

$$
\text{output}[i][j] = \log\left(\frac{e^{\text{input}[i][j]}}{\sum_{k=0}^{N-1} e^{\text{input}[i][k]}}\right)
$$

which simplifies to:

$$
\text{output}[i][j] = \text{input}[i][j] - \log\left(\sum_{k=0}^{N-1} e^{\text{input}[i][k]}\right)
$$

## Input:
- Matrix `input` of size $M \times N$ containing floating-point values

## Output:
- Matrix `output` of size $M \times N$ containing the log-softmax values

## Notes:
- Both matrices are stored in row-major order
- Log-Softmax is applied independently along each row (dim=1)
- For numerical stability, consider using the identity: $\log\text{softmax}(x_i) = x_i - \max(x) - \log\sum_k e^{x_k - \max(x)}$
- Log-Softmax is commonly used as the final layer in classification models paired with Negative Log-Likelihood loss
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/24_LogSoftmax.py)