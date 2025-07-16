---
slug: "linear-sub-mul-relu"
title: "Linear Transformation with Subtraction, Multiplication, and ReLU"
difficulty: "MEDIUM"
author: "sarthak"
tags: ["linear", "activation-function", "neural-network"]
parameters:
  - name: "input"
    type: "[VAR]"
    pointer: "true"
    const: "true"
  
  - name: "weights"
    type: "[VAR]"
    pointer: "true"
    const: "true"
    
  - name: "bias"
    type: "[VAR]"
    pointer: "true"
    const: "true"
    
  - name: "subtract_value"
    type: "float"
    pointer: "false"
    const: "false"
    
  - name: "multiply_value"
    type: "float"
    pointer: "false"
    const: "false"

  - name: "output"
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "batch_size" 
    type: "size_t"
    pointer: "false"
    constant: "false"
    
  - name: "in_features"
    type: "size_t"
    pointer: "false"
    constant: "false"
    
  - name: "out_features"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Perform a linear transformation followed by subtraction, multiplication, and ReLU activation:

1. **Linear Transformation**: $Y = XW^T + b$
2. **Subtraction**: $Z = Y - s$
3. **Multiplication**: $A = Z \cdot m$
4. **ReLU Activation**: $\text{output} = \max(0, A)$

Where:
- $X$ is the input matrix of shape $(B, N)$ (batch_size, input_features)
- $W$ is the weight matrix of shape $(M, N)$ (output_features, input_features)
- $b$ is the bias vector of shape $(M)$ (output_features)
- $s$ is the subtract_value scalar
- $m$ is the multiply_value scalar

The complete operation can be expressed as:
$$
\text{output} = \text{ReLU}((XW^T + b - s) \cdot m)
$$

## Input:
- Matrix $X$ of size $B \times N$ (input features)
- Matrix $W$ of size $M \times N$ (weight matrix)
- Vector $b$ of size $M$ (bias vector)
- Scalar $s$ (subtract_value)
- Scalar $m$ (multiply_value)

## Output:
- Matrix $\text{output}$ of size $B \times M$ (transformed and activated features)

## Notes:
- All matrices are stored in row-major order
- This operation is common in neural networks where additional scaling and shifting is applied after linear transformations
- The ReLU activation ensures non-negative outputs: $\text{ReLU}(x) = \max(0, x)$
- Example use case: Custom neural network layers with learnable scaling and shifting parameters 