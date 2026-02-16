---
slug: "edge-detect"
title: "Edge Detection"
difficulty: "EASY"
author: "sarthak"
tags: ["graphics"]
parameters:
  - name: "input_image"
    type: "float"
    pointer: "true"
    const: "true"
  
  - name: "output_image" 
    type: "float"
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

---

Detect edges in a grayscale image using simple gradient-based edge detection:

$$
G_x[i][j] = \frac{\text{Input}[i][j+1] - \text{Input}[i][j-1]}{2}
$$

$$
G_y[i][j] = \frac{\text{Input}[i+1][j] - \text{Input}[i-1][j]}{2}
$$

$$
\text{Output}[i][j] = \sqrt{G_x[i][j]^2 + G_y[i][j]^2}
$$

The result is normalized to the range [0, 255].

This algorithm computes horizontal and vertical gradients, then combines them to find edge strength at each pixel.

## Input:
- Grayscale image of size $\text{height} \times \text{width}$

## Output:
- Edge detected image of size $\text{height} \times \text{width}$

## Notes:
- The input tensor is a single-channel grayscale image
- Only compute gradients for interior pixels (ignore 1-pixel border)
- Border pixels remain zero in the output
