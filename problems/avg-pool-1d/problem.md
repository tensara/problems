---
slug: "avg-pool-1d"
title: "1D Average Pooling"
difficulty: "EASY" 
author: "sarthak"
tags: ["pooling"]
parameters:
  - name: "input"
    type: "float"
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
    type: "float"
    pointer: "true"
    const: "false"

  - name: "H"
    type: "size_t"
    pointer: "false"
    constant: "false"
    
---

Perform 1D average pooling on an input tensor:
$$
\text{output}[i] = \frac{1}{k}\sum_{m=0}^{k-1} \text{input}[S \cdot i + m - P]
$$

The average pooling operation slides a window of size $k$ over the input tensor with stride $S$ and padding $P$, computing the average value within each window position.

## Input:
- Matrix `input` of size $\text{H}$ (input tensor)
- `kernel_size` ($k$): Size of the pooling window
- `stride` ($S$): Step size between window positions
- `padding` ($P$): Number of zero-padding elements added on all sides

## Output:
- Matrix `output` of size $\text{H}_{\text{out}}$ where:
  $$\text{H}_{\text{out}} = \left\lfloor\frac{\text{H} + 2P - k}{S} + 1\right\rfloor$$

## Notes:
- Zero padding is applied when specified by the padding parameter
- For values outside the input boundaries (after padding), use zero values in the average computation
- The denominator ($k$) should always be the full kernel size, even when some elements are outside the input boundaries
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/44_Average_Pooling_1D.py)