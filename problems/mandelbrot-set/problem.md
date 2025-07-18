---
slug: "mandelbrot-set"
title: "Mandelbrot Set"
difficulty: "HARD"
author: "sarthak"
tags: ["fractals", "complex-numbers", "visualization"]
parameters:
  - name: "output_image"
    type: "[VAR]"
    pointer: "true"
    const: "false"
  
  - name: "real_min"
    type: "float"
    pointer: "false"
    constant: "false"
    
  - name: "real_max"
    type: "float"
    pointer: "false"
    constant: "false"
    
  - name: "imag_min"
    type: "float"
    pointer: "false"
    constant: "false"
    
  - name: "imag_max"
    type: "float"
    pointer: "false"
    constant: "false"
    
  - name: "width"
    type: "int"
    pointer: "false"
    constant: "false"
    
  - name: "height"
    type: "int"
    pointer: "false"
    constant: "false"
    
  - name: "max_iter"
    type: "int"
    pointer: "false"
    constant: "false"

---

Compute the Mandelbrot set, a famous fractal defined by the iterative complex function $z_{n+1} = z_n^2 + c$.

For each point $c$ in the complex plane, the Mandelbrot set determines whether the sequence:
$$z_0 = 0, \quad z_{n+1} = z_n^2 + c$$
remains bounded or diverges to infinity.

## Algorithm:

1. Map each pixel $(x, y)$ to a complex number $c = x_{real} + i \cdot y_{imag}$ in the specified region
2. Initialize $z = 0$
3. Iterate $z = z^2 + c$ up to `max_iter` times
4. If $|z| > 2$ at any point, the sequence diverges (store iteration count)
5. If $|z| \leq 2$ after `max_iter` iterations, assume the point is in the set (store `max_iter`)

## Mathematical Details:

The complex number arithmetic for $z^2 + c$ where $z = a + bi$ and $c = p + qi$:
- $z^2 = (a + bi)^2 = a^2 - b^2 + 2abi$
- $z^2 + c = (a^2 - b^2 + p) + (2ab + q)i$

The divergence condition $|z| > 2$ is equivalent to $a^2 + b^2 > 4$ (avoiding square root).

## Coordinate Mapping:

For a pixel at position $(x, y)$ in an image of size $width \times height$:
- $c_{real} = real_{min} + x \cdot \frac{real_{max} - real_{min}}{width}$
- $c_{imag} = imag_{min} + y \cdot \frac{imag_{max} - imag_{min}}{height}$

## Input:
- Complex plane bounds: `real_min`, `real_max`, `imag_min`, `imag_max`
- Image dimensions: `width`, `height`
- Maximum iterations: `max_iter`

## Output:
- 2D array of iteration counts (height × width)
- Value at each pixel represents when the sequence diverged
- Points in the set have value `max_iter`

## GPU Optimization:
- Each pixel can be computed completely independently
- No shared memory or synchronization needed
- Excellent candidate for embarrassingly parallel computation
- Consider early termination when divergence is detected

## Visualization Notes:
- Low iteration counts (early divergence) typically represent areas outside the set
- High iteration counts represent points close to or inside the set
- The boundary creates the characteristic fractal patterns
- Different color mappings can highlight different aspects of the fractal 