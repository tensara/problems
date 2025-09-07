---
slug: "histogram"
title: "Image Histogram"
difficulty: "EASY"
author: "sarthak"
tags: ["graphics", "statistics"]
parameters:
  - name: "image"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "num_bins" 
    type: "int"
    pointer: "false"
    constant: "false"
  
  - name: "histogram" 
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

---

Compute the histogram of a grayscale image by counting the frequency of each pixel intensity:

$$
\text{Histogram}[k] = \sum_{i=0}^{H-1} \sum_{j=0}^{W-1} \mathbf{1}_{\{\text{Input}[i][j] = k\}}
$$

where $\mathbf{1}_{\{\cdot\}}$ is the indicator function that equals 1 when the condition is true, 0 otherwise.

This creates a frequency distribution showing how many pixels have each intensity value.

## Input:
- Grayscale image of size $\text{height} \times \text{width}$
- Number of histogram bins (typically 64, 128, or 256)

## Output:
- Histogram array of size $\text{num\_bins}$ containing pixel counts

## Notes:
- The input tensor contains integer pixel values in range [0, num_bins-1]
- Each histogram bin counts pixels with that specific intensity value
- The sum of all histogram bins equals the total number of pixels