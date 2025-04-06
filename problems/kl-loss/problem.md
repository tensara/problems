---
slug: "kl-loss"
title: "Kullback-Leibler Divergence"
difficulty: "MEDIUM"
author: "sarthak"
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

  - name: "n"
    type: "size_t"
    pointer: "false"
    constant: "false"

---

Compute the element-wise Kullback-Leibler Divergence between two probability distributions, `predictions` and `targets`.

The Kullback-Leibler Divergence is a measure of how one probability distribution diverges from a second, expected probability distribution. For discrete probability distributions P and Q, the KL divergence is defined as:

$$
D_{KL}(P || Q) = \sum_{i} P(i) \log\left(\frac{P(i)}{Q(i)}\right) = \sum_{i} P(i) (\log P(i) - \log Q(i))
$$

In this problem, you will compute the **element-wise** KL divergence *before* the summation step. That is, for each element:

$$
\text{output}[i] = \text{targets}[i] \cdot (\log(\text{targets}[i]) - \log(\text{predictions}[i]))
$$

Note that when `targets[i]` is 0, the contribution to the KL divergence is 0 (by convention, using the limit $\lim_{x \to 0} x \log(x) = 0$).

## Input:
- Tensor `predictions` of size $N$ representing a probability distribution Q (all values > 0 and sum to 1)
- Tensor `targets` of size $N$ representing a probability distribution P (all values ≥ 0 and sum to 1)

## Output:
- Tensor `output` of size $N$, where `output[i]` contains the element-wise KL divergence contribution.

## Notes:
- All tensors are flat 1D arrays (or treated as such) and stored contiguously in memory.
- You should handle the case where `targets[i]` is 0 correctly (the contribution should be 0).
- To avoid numerical issues, you should add a small epsilon (e.g., 1e-10) to predictions and targets before computing logarithms.
- The full KL divergence can be computed by summing all elements of the output tensor.
