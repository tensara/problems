---
slug: "cross-entropy"
title: "Cross Entropy"
difficulty: "EASY"
author: "sarthak"
tags: ["loss-function"]
parameters:
  - name: "predictions"
    type: "[VAR]"
    pointer: "true"
    const: "true"
  
  - name: "targets"
    type: "int"
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
  
  - name: "c"
    type: "size_t"
    pointer: "false"
    constant: "false"

---

Compute the cross entropy loss between predictions and targets for multi-class classification tasks.

The Cross Entropy Loss is a common loss function used in classification problems. For a single sample, it is defined as:
$$
\text{cross\_entropy}(x, y) = -\sum_{c=1}^{C} y_c \log(p_c)
$$

where $x$ represents the predicted logits, $y$ represents the target class (usually one-hot encoded), $C$ is the number of classes, and $p_c$ is the softmax probability for class $c$:
$$
p_c = \frac{e^{x_c}}{\sum_{j=1}^{C} e^{x_j}}
$$

In practice, for numerical stability, we often compute the loss using the log-softmax operation:
$$
\text{cross\_entropy}(x, y) = -\sum_{c=1}^{C} y_c \log(\text{softmax}(x)_c)
$$

This problem asks you to compute the cross entropy loss for each sample in a batch.

## Input:
- Tensor `predictions` of size $N \times C$ (N samples, C classes)
- Tensor `targets` of size $N$ (class indices, integers in range [0, C-1])

## Output:
- Tensor `output` of size $N$, where `output[i]` contains the cross entropy loss for the i-th sample.

## Notes:
- All tensors are stored contiguously in memory.
- The implementation should be numerically stable.
- For targets with only one class label per sample, the target tensor contains class indices rather than one-hot encoded vectors.
