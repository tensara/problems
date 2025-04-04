---
slug: "max-pool-2d"
title: "2D Max Pooling"
difficulty: "MEDIUM" 
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

  - name: "dilation"
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
  
---

Perform 2D max pooling on an input tensor:
$$
\text{output}[i,j] = \max_{m=0,n=0}^{k-1,k-1} \text{input}[S \cdot i + D \cdot m - P, S \cdot j + D \cdot n - P]
$$

The max pooling operation slides a window of size $k \times k$ over the input tensor with stride $S$, dilation $D$, and padding $P$, computing the maximum value within each window position.

## Input:
- Matrix `input` of size $\text{H} \times \text{W}$ (input tensor)
- `kernel_size` ($k$): Size of the pooling window
- `stride` ($S$): Step size between window positions
- `padding` ($P$): Number of zero-padding elements added on all sides
- `dilation` ($D$): Spacing between kernel elements

## Output:
- Matrix `output` of size $\text{H}_{\text{out}} \times \text{W}_{\text{out}}$ where:
  $$\text{H}_{\text{out}} = \left\lfloor\frac{\text{H} + 2P - D(k-1) - 1}{S}\right\rfloor + 1$$
  $$\text{W}_{\text{out}} = \left\lfloor\frac{\text{W} + 2P - D(k-1) - 1}{S}\right\rfloor + 1$$

## Notes:
- All matrices are stored in row-major order
- Zero padding is applied when specified by the padding parameter
- For values outside the input boundaries (after padding), use negative infinity
- Dilation controls the spacing between kernel elements, creating an effective kernel size of $D(k-1) + 1$
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/42_Max_Pooling_2D.py)