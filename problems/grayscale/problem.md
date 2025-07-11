---
slug: "grayscale"
title: "Grayscale Conversion"
difficulty: "EASY"
author: "sarthak"
tags: ["graphics"]
parameters:
  - name: "rgb_image"
    type: "[VAR]"
    pointer: "true"
    const: "true"
  
  - name: "grayscale_output" 
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

  - name: "channels" 
    type: "size_t"
    pointer: "false"
    constant: "false"

---

Perform RGB to grayscale conversion on an input image using the weighted method:
$$
\text{Gray}[i][j] = 0.299 \cdot \text{R}[i][j] + 0.587 \cdot \text{G}[i][j] + 0.114 \cdot \text{B}[i][j]
$$

This formula accounts for human perception of color, with green contributing most to the intensity perceived by humans.

## Input:
- RGB image of size $\text{height} \times \text{width} \times \text{3}$

## Output:
- Grayscale image of size $\text{height} \times \text{width}$

## Notes:
- The input tensor is in HWC format (height, width, channels)
- In memory, the tensor is stored in row-major order with interleaved channels (R,G,B,R,G,B,...)
- Each pixel has 3 channels in RGB order
- Pixel values are in the range [0, 255] for both input and output
- The output is a single-channel grayscale image 