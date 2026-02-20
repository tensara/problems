---
slug: "matmul-swish"
title: "Matrix Multiplication with Swish Activation"
difficulty: "MEDIUM" 
author: "sarthak"
tags: ["matmul", "activation-function", "fused"]
---

Perform matrix multiplication followed by Swish activation and scaling:
$$
\text{output} = \text{scaling\_factor} \cdot (\text{input} \cdot \text{weight}^T + \text{bias}) \cdot \sigma((\text{input} \cdot \text{weight}^T + \text{bias}))
$$

where $\sigma(x)$ is the sigmoid function:
$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

The operation consists of three main steps:
1. Linear transformation: $z = \text{input} \cdot \text{weight}^T + \text{bias}$
2. Swish activation: $\text{swish}(z) = z \cdot \sigma(z)$
3. Scaling: $\text{output} = \text{scaling\_factor} \cdot \text{swish}(z)$

## Input:
- Matrix `input_matrix` of size $\text{batch\_size} \times \text{in\_features}$
- Matrix `weight_matrix` of size $\text{out\_features} \times \text{in\_features}$
- Vector `bias` of size $\text{out\_features}$
- Scalar `scaling_factor` for final scaling

## Output:
- Matrix `output` of size $\text{batch\_size} \times \text{out\_features}$

## Notes:
- All matrices are stored in row-major order
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/59_Matmul_Swish_Scaling.py)