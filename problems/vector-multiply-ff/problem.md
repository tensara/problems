---
slug: "vector-multiply-ff"
title: "Vector Multiplication over Finite Field"
difficulty: "MEDIUM"
author: "soham"
tags: ["crypto"]
---

Perform element-wise multiplication of two vectors in the finite field:

$$
c_i = (a_i \cdot b_i) \bmod p
$$

Where the modulus is the 31-bit Mersenne prime:

$$
p = 2^{31} - 1 = 2147483647
$$

## Input
- Vectors `a` and `b` of length $n$, with each element in $[0, p)$.

## Output
- Vector `c` of length $n$ such that:
  $$
  c_i = a_i \cdot b_i \bmod p
  $$
  
## Notes
- $1 \le n \le 2^{30}$
- Inputs and outputs are 32-bit unsigned integers.
- Intermediate products must be reduced modulo $p$.
- A simple (correct) implementation:
  ```cpp
  const uint32_t P = 2147483647u;
  uint64_t prod = (uint64_t)a[i] * (uint64_t)b[i];
  c[i] = (uint32_t)(prod % P);
  ```
