---
slug: "bilateral-filter"
title: "Bilateral Filter"
difficulty: "HARD"
author: "sarthak"
tags: ["graphics", "filtering", "advanced"]
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

  - name: "sigma_spatial"
    type: "float"
    pointer: "false"
    constant: "false"

  - name: "sigma_color"
    type: "float"
    pointer: "false"
    constant: "false"

  - name: "kernel_size"
    type: "int"
    pointer: "false"
    constant: "false"

---

Implement a bilateral filter for edge-preserving image smoothing. The bilateral filter reduces noise while preserving sharp edges by considering both spatial proximity and color similarity.

The bilateral filter weight for a pixel at position $(i, j)$ relative to the center pixel at $(x, y)$ is:

$$
w(i, j, x, y) = \exp\left(-\frac{(i-x)^2 + (j-y)^2}{2\sigma_s^2}\right) \cdot \exp\left(-\frac{(I(i,j) - I(x,y))^2}{2\sigma_c^2}\right)
$$

The filtered output at position $(x, y)$ is:

$$
I_f(x, y) = \frac{\sum_{i,j \in \Omega} w(i, j, x, y) \cdot I(i, j)}{\sum_{i,j \in \Omega} w(i, j, x, y)}
$$

Where:
- $\Omega$ represents the kernel neighborhood around pixel $(x, y)$
- $\sigma_s$ is the spatial standard deviation (controls spatial smoothing)
- $\sigma_c$ is the color standard deviation (controls edge preservation)
- $I(i, j)$ is the intensity at pixel $(i, j)$

## Input:
- Grayscale image of size $\text{height} \times \text{width}$
- Spatial standard deviation $\sigma_s$ 
- Color standard deviation $\sigma_c$
- Kernel size (must be odd)

## Output:
- Filtered grayscale image of size $\text{height} \times \text{width}$

## Algorithm:
1. For each pixel $(x, y)$ in the output image:
   - Initialize weighted sum = 0, weight sum = 0
   - For each pixel $(i, j)$ in the kernel neighborhood:
     - Calculate spatial weight using Gaussian of distance
     - Calculate color weight using Gaussian of intensity difference
     - Multiply weights and accumulate
   - Set output pixel = weighted sum / weight sum

## Notes:
- Handle boundary conditions by clamping coordinates to image bounds
- The spatial kernel follows a Gaussian distribution based on pixel distance
- The color kernel follows a Gaussian distribution based on intensity difference
- This is computationally intensive but highly parallelizable on GPUs
- Each output pixel can be computed independently 