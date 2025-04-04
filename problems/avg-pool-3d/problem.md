---
slug: "avg-pool-3d"
title: "3D Average Pooling"
difficulty: "HARD" 
author: "sarthak"
tags: ["pooling"]
parameters:
  - name: "input"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "kernel_size"
    type: "int"
    pointer: "false"
    constant: "false"
    
  - name: "stride" 
    type: "int"
    pointer: "false"
    constant: "false"

  - name: "padding"
    type: "int"
    pointer: "false"
    constant: "false"

  - name: "output" 
    type: "[VAR]"
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

  - name: "D"
    type: "size_t"
    pointer: "false"
    constant: "false"
  
---

Perform 3D average pooling on an input tensor:
$$
\text{output}[i,j,k] = \frac{1}{k^3}\sum_{m=0}^{k-1}\sum_{n=0}^{k-1}\sum_{o=0}^{k-1} \text{input}[S \cdot i + m - P, S \cdot j + n - P, S \cdot k + o - P]
$$

The average pooling operation slides a window of size $k \times k \times k$ over the input tensor with stride $S$ and padding $P$, computing the average value within each window position.

## Input:
- Matrix `input` of size $\text{H} \times \text{W} \times \text{D}$ (input tensor)
- `kernel_size` ($k$): Size of the pooling window
- `stride` ($S$): Step size between window positions
- `padding` ($P$): Number of zero-padding elements added on all sides

## Output:
- Matrix `output` of size $\text{H}_{\text{out}} \times \text{W}_{\text{out}} \times \text{D}_{\text{out}}$ where:
  $$\text{H}_{\text{out}} = \left\lfloor\frac{\text{H} + 2P - k + 1}{S}\right\rfloor$$
  $$\text{W}_{\text{out}} = \left\lfloor\frac{\text{W} + 2P - k + 1}{S}\right\rfloor$$
  $$\text{D}_{\text{out}} = \left\lfloor\frac{\text{D} + 2P - k + 1}{S}\right\rfloor$$

## Notes:
- All tensors are stored in row-major order
- Zero padding is applied when specified by the padding parameter
- For values outside the input boundaries (after padding), use zero values in the average computation
- The denominator ($k^3$) should always be the full kernel size, even when some elements are outside the input boundaries
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/46_Average_Pooling_3D.py)