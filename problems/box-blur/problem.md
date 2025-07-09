---
slug: "box-blur"
title: "Box Blur"
difficulty: "EASY"
author: "sarthak"
tags: ["graphics", "convolution"]
parameters:
  - name: "input_image"
    type: "[VAR]"
    pointer: "true"
    const: "true"
  
  - name: "output_image" 
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

  - name: "kernel_size" 
    type: "int"
    pointer: "false"
    constant: "false"

---

Apply a box blur filter to a grayscale image by averaging pixels in a square neighborhood:

$$
\text{Output}[i][j] = \frac{1}{N} \sum_{u=-k}^{k} \sum_{v=-k}^{k} \text{Input}[i+u][j+v]
$$

where $k = \lfloor \text{kernel\_size}/2 \rfloor$ and $N$ is the number of valid pixels in the kernel.

This creates a blurring effect by smoothing out pixel values. The larger the kernel size, the more blurred the result.

## Input:
- Grayscale image of size $\text{height} \times \text{width}$
- Kernel size (odd integer, e.g., 3, 5, 7)

## Output:
- Blurred image of size $\text{height} \times \text{width}$

## Notes:
- The input tensor is a single-channel grayscale image
- Handle edge cases by only averaging available pixels (no padding)
- For pixels near the border, use a smaller effective kernel
- The kernel size should be odd (3x3, 5x5, 7x7, etc.)
- This is a fundamental operation in computer graphics and image processing 