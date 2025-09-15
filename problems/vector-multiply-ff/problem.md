---
slug: "vector-multiply-babybear"
title: "Vector Multiplication over BabyBear Field"
difficulty: "MEDIUM"
author: "soham"
tags: ["finite-field", "modular-arithmetic", "vector"]
parameters:
  - name: "d_input1"
    type: "uint32_t"
    pointer: true
    const: true

  - name: "d_input2"
    type: "uint32_t"
    pointer: true
    const: true

  - name: "d_output"
    type: "uint32_t"
    pointer: true
    const: false

  - name: "n"
    type: "size_t"
    pointer: false
    constant: false
---

Perform element-wise multiplication of two vectors in the BabyBear finite field:
$$
c_i = (a_i \cdot b_i) \mod p
$$

Where the modulus is:
$$
p = 2^{31} - 2^{27} + 1 = 0x78000001
$$

---

## Input

- Vectors `a` and `b` of length `n`, with each element in `[0, p)`.

## Output

- Vector `c` of length `n` such that:
$$
c_i = a_i \cdot b_i \mod p
$$

---

## Constraints

- `1 <= n <= 2^30`
- Inputs and outputs are 32-bit unsigned integers.
- Overflow is expected; you must reduce modulo `p` correctly.

---

## Baseline Implementation

A simple implementation might look like:

```cpp
uint64_t prod = (uint64_t)a[i] * b[i];
c[i] = prod % 0x78000001;
```

This is correct but slow — `%` is expensive on GPU.

---

## Optimizations to Explore

- Use **Montgomery** or **Barrett reduction** to avoid `%` ops
- Inline your `mod` and `mul` helpers with `__forceinline__`
- Use `registers` to hold constants
- **Fuse multiply and reduce** into a single operation
- Coalesce memory accesses to avoid warp divergence
- Use appropriate block/thread sizing to maximize occupancy

---

## Verification

Your solution must return results identical to the reference CPU-side or PyTorch mod logic.

---


