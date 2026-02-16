---
slug: "cosine-similarity"
title: "Cosine Similarity"
difficulty: "EASY"
author: "sarthak"
tags: ["loss-function"]
parameters:
  - name: "predictions"
    type: "float"
    pointer: "true"
    const: "true"
  
  - name: "targets"
    type: "float"
    pointer: "true"
    const: "true"

  - name: "output" 
    type: "float"
    pointer: "true"
    const: "false"

  - name: "n"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "d"
    type: "size_t"
    pointer: "false"
    constant: "false"

---

Compute the element-wise cosine similarity between two input tensors, `predictions` and `targets`.

The Cosine Similarity between two vectors measures the cosine of the angle between them. It is defined as:
$$
\text{cosine\_similarity}(x, y) = \frac{x \cdot y}{\max(\epsilon, ||x||) \cdot \max(\epsilon, ||y||)}
$$

where $x$ represents predictions, $y$ represents targets, $x \cdot y$ is the dot product, $||x||$ is the L2 norm of $x$, $||y||$ is the L2 norm of $y$, and $\epsilon = 10^{-8}$ is a small constant to prevent division by zero.

This problem asks you to compute the negative cosine similarity for each pair of vectors in `predictions` and `targets`. 
The formula for the negative cosine similarity is:
$$
\text{loss}(x, y) = 1 - \text{cosine\_similarity}(x, y)
$$

## Input:
- Tensor `predictions` of size $N \times D$ (N vectors of dimension D)
- Tensor `targets` of size $N \times D$ (N vectors of dimension D)

## Output:
- Tensor `output` of size $N$, where `output[i]` contains the negative cosine similarity between the i-th prediction and target vectors.

## Notes:
- To handle numerical stability, use $\epsilon = 10^{-8}$ in the denominator as shown in the formula above
- The norm of a vector is calculated as the square root of the sum of squares of its elements
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/97_CosineSimilarityLoss.py)