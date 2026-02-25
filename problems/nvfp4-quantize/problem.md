---
slug: "nvfp4-quantize"
title: "NVFP4 Quantization"
difficulty: "MEDIUM"
author: "sarthak"
tags: ["quantization", "nvfp4"]
gpus: ["B200"]
---

Quantize an input FP16 matrix into the [NVFP4 format](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/). It uses a two-level scaling strategy: a global scale moves the entire tensor into the representable range of a block (FP4 $\times$ FP8), then a local per-block scale moves each block into FP4 range. A rough outline is to:

- Divide the `[M, K]` matrix into contiguous blocks of 16 elements along K.
- For each block $b$, find the block (absolute) max and normalize to FP4 range:
$$
s_{\text{dec},b} = \frac{\text{amax}_b}{6}
$$
- Multiply by the global encode scale and cast to E4M3 via round to nearest even (these FP8 values are what get stored in the `scale` output):
$$
s_{\text{dec},b,\text{e4m3}} = \texttt{e4m3}(s_{\text{dec},b} \cdot s_{\text{enc}})
$$
- Invert the quantized decode scale (cast back to fp32) and divide by the global decode scale ($s_{\text{dec}} = 1 / s_{\text{enc}}$):
$$
s_{\text{enc},b} = \frac{1}{\texttt{fp32}(s_{\text{dec},b,\text{e4m3}}) \cdot s_{\text{dec}}}
$$

Now, you can quantize to FP4 E2M1:
- For each element $x_i$ in the block: $\hat{x}_i = q(x_i \cdot s_{\text{enc},b})$, where $q(\cdot)$ is FP4 quantization.
- Finally, pack pairs of adjacent E2M1 values into single bytes (two 4-bit values per uint8).

## Input
- $a$: `fp16` pointer to row-major tensor of shape $M \times K$ 
- $sf_g$: `fp32` scalar, the global scale factor defined as:

$$
\frac{\texttt{FP4\_AMAX} \times \texttt{FP8\_AMAX}}{\text{amax}(|a|)} = \frac{6 \times 448}{\text{amax}(|a|)}
$$

- $M$, $K$: dimensions of $a$

## Output
- $q$: `uint8` pointer, packed E2M1 quantized values of shape $M \times K/2$
- $scale$: `uint8` pointer, FP8 E4M3 per-block scale factors in the swizzled 128x4 layout (see below)


## Notes
Instead of storing the scale factors in naive row-major order, they must be arranged in a swizzled layout for tensor core consumption.

To do this, we first tile the 2D array into 128-row $\times$ 4-column tiles (pad M to a multiple of 128, this will be needed to pass the sample). Then, _within_ each 128-row M-tile, reorder the 128 rows as a 32 $\times$ 4 column-major block. That is, rows 0..31 go first, then 32..63, 64..95, 96..127, but interleaved column-first so that rows 32 apart in logical space become adjacent in memory. Thus, the memory order is: 0, 32, 64, 96, 1, 33, 65, 97, etc.  Check out the [cuBLAS 1D Block Scaling Factors Layout](https://docs.nvidia.com/cuda/cublas/#d-block-scaling-factors-layout) documentation for more info.

We use FlashInfer's [nvfp4_quantize](https://docs.flashinfer.ai/generated/flashinfer.fp4_quantization.nvfp4_quantize.html) with `SfLayout.layout_128x4` (the default layout) as the ground truth. Submissions are validated by dequantizing both the reference and submitted outputs via [e2m1_and_ufp8sf_scale_to_float](https://docs.flashinfer.ai/generated/flashinfer.fp4_quantization.e2m1_and_ufp8sf_scale_to_float.html) and checking closeness.
