---
slug: "box-blur"
title: "Box Blur"
difficulty: "EASY"
author: "sarthak"
tags: ["graphics", "convolution"]
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
