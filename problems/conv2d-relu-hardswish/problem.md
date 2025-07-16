---
slug: "conv2d-relu-hardswish"
title: "2D Convolution + ReLU + HardSwish Fusion"
difficulty: "HARD"
author: "sarthak"
tags: ["conv2d", "relu", "hardswish", "fusion"]
parameters:
  - name: "input_image"
    type: "[VAR]"
    pointer: "true"
    const: "true"
  
  - name: "kernel"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "output" 
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "height"
    type: "size_t"
    pointer: "false"
    constant: "false"
    
  - name: "width" 
    type: "size_t"
    pointer: "false"
    constant: "false"
    
  - name: "kernel_height"
    type: "size_t"
    pointer: "false"
    constant: "false"
    
  - name: "kernel_width"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Perform fused 2D convolution followed by ReLU activation followed by HardSwish activation:

1. **2D Convolution**: 
   $$C[i][j] = \sum_{u=0}^{K_h-1} \sum_{v=0}^{K_w-1} I[i+u-\frac{K_h-1}{2}][j+v-\frac{K_w-1}{2}] \cdot K[u][v]$$

2. **ReLU Activation**:
   $$R[i][j] = \max(0, C[i][j])$$

3. **HardSwish Activation**:
   $$O[i][j] = R[i][j] \cdot \frac{\text{ReLU6}(R[i][j] + 3)}{6}$$

where $\text{ReLU6}(x) = \min(6, \max(0, x))$.

## Input
- Input image $I$ of size $H \times W$
- Convolution kernel $K$ of size $K_h \times K_w$ (both dimensions must be odd)

## Output
- Output image $O$ of size $H \times W$

## Notes:
- Zero padding is applied to maintain the same output size as input
- The kernel dimensions must be odd numbers
- HardSwish is an efficient approximation of Swish activation
- The fusion of these operations can significantly reduce memory bandwidth requirements
- Consider optimizing memory access patterns across all three operations 