---
slug: "conv2d-relu-hardswish"
title: "2D Convolution with ReLU and HardSwish"
difficulty: "MEDIUM"
author: "sarthak"
tags: ["convolution", "activation-function", "fused"]
parameters:
  - name: "image"
    type: "float"
    pointer: "true"
    const: "true"
  
  - name: "kernel"
    type: "float"
    pointer: "true"
    const: "true"

  - name: "output" 
    type: "float"
    pointer: "true"
    const: "false"

  - name: "H"
    type: "size_t"
    pointer: "false"
    constant: "false"
    
  - name: "W" 
    type: "size_t"
    pointer: "false"
    constant: "false"
    
  - name: "Kh"
    type: "size_t"
    pointer: "false"
    constant: "false"
    
  - name: "Kw"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Perform a 2D convolution followed by ReLU activation followed by HardSwish activation:

1. **2D Convolution**: 
   $$C[i][j] = \sum_{u=0}^{K_h-1} \sum_{v=0}^{K_w-1} I\left[i+u-\frac{K_h-1}{2}\right]\left[j+v-\frac{K_w-1}{2}\right] \cdot K[u][v]$$

2. **ReLU Activation**:
   $$R[i][j] = \max(0, C[i][j])$$

3. **HardSwish Activation**:
   $$O[i][j] = R[i][j] \cdot \frac{\text{ReLU6}(R[i][j] + 3)}{6}$$

where $\text{ReLU6}(x) = \min(6, \max(0, x))$.

## Input
- `image` of size $H \times W$
- `kernel` of size $K_h \times K_w$ (both dimensions must be odd)

## Output
- `output` of size $H \times W$

## Notes:
- Use zero padding at the boundaries where the kernel extends beyond the input image
- Both kernel height $K_h$ and kernel width $K_w$ will be odd integers