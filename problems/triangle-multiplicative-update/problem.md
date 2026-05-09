---
slug: "triangle-multiplicative-update"
title: "Triangle Multiplicative Update"
difficulty: "HARD"
author: "josusanmartin"
tags: ["alphafold", "matmul", "tensor-contraction", "bio-ml"]
---

Implement the core outgoing Triangle Multiplicative Update contraction used by AlphaFold-style pair representations.

This problem is based on GPU MODE's [(Mini) Competition #3: AlphaFold's Triangle Multiplicative Update](https://stormy-sailor-96a.notion.site/GPU-MODE-Mini-Competition-3-AlphaFold-s-Triangle-Multiplicative-Update-207221cc2ffa8034b3eddff1d898dc14). The Tensara version isolates the cubic contraction from the full AlphaFold block so CUDA submissions can focus on the memory-layout and tensor-contraction challenge.

Given two transformed pair tensors:

$$
L, R \in \mathbb{R}^{B \times N \times N \times H}
$$

and a pair mask:

$$
M \in \mathbb{R}^{B \times N \times N},
$$

compute:

$$
O[b, i, j, h] =
\sum_{k=0}^{N-1}
\left(L[b, i, k, h] \cdot M[b, i, k]\right)
\left(R[b, j, k, h] \cdot M[b, j, k]\right).
$$

This is equivalent to the outgoing TriMul einsum:

```python
output = einsum("bik h,bjk h->bij h", left * mask[..., None], right * mask[..., None])
```

where spacing is added only for readability.

## Input

- Tensor `left` of shape $B \times N \times N \times H$
- Tensor `right` of shape $B \times N \times N \times H$
- Tensor `mask` of shape $B \times N \times N$, with entries equal to `0` or `1`

## Output

- Tensor `output` of shape $B \times N \times N \times H$

## Notes

- All tensors are stored in row-major order.
- The last dimension `H` is contiguous in memory.
- The contraction reduces over the middle sequence index `k`.
- The mask is always provided. No-mask cases use an all-ones mask.
- Projection, sigmoid gates, and layer normalization from the full GPU MODE problem are intentionally excluded here so the Tensara task stays focused on the hard memory-layout and tensor-contraction part.
- Using Tensor Cores directly is non-trivial because the matrices for each hidden channel are strided by `H` in the input layout.

## Test Case Sizes

- B=1, N=128, H=64, normal input, no mask
- B=1, N=256, H=128, normal input, random mask
- B=2, N=256, H=128, clamped Cauchy input, random mask
- B=1, N=512, H=128, normal input, no mask
- B=1, N=1024, H=128, normal input, random mask
