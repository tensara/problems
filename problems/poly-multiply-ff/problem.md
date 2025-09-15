---
slug: "poly-multiply-ff"
title: "Polynomial Multiplication over Finite Field"
difficulty: "MEDIUM"
author: "soham"
tags: ["finite-field", "polynomial", "modular-arithmetic"]
parameters:
  - name: "d_input1"
    type: "uint64_t"
    pointer: true
    const: true

  - name: "d_input2"
    type: "uint64_t"
    pointer: true
    const: true

  - name: "d_output"
    type: "uint64_t"
    pointer: true
    const: false

  - name: "n"
    type: "size_t"
    pointer: false
    constant: false
---

Multiply two polynomials over the Goldilocks field:
Use the prime modulus
\[
p \;=\; 2^{64} - 2^{32} + 1 \;=\; 18446744069414584321.
\]

Let \(a(x)\) and \(b(x)\) be polynomials of degree \(n-1\) represented by coefficient vectors of length \(n\):
\[
a(x) = \sum*{i=0}^{n-1} a_i x^i, \qquad
b(x) = \sum*{j=0}^{n-1} b_j x^j,
\]
with all coefficients in \([0, p)\).

Compute their product modulo \(p\):
\[
c(x) \;=\; a(x)\cdot b(x) \pmod p,
\]
and return the coefficient vector of \(c(x)\) of length \(2n - 1\):
\[
c*k \;=\; \sum*{i+j=k} a_i\cdot b_j \;\bmod p.
\]

---

## Input

- Two device arrays `d_input1`, `d_input2` of length `n`, elements in \([0, p)\) as `uint64_t`.
- `n` is a power of two (e.g., 1024, 4096, …).

## Output

- Device array `d_output` of length `2n - 1` with the coefficients of \(c(x)\) modulo \(p\).

---

## Example (mod small p just for illustration)

```
a = [1, 2, 3]
b = [4, 5, 6]
→ c = [4, 13, 28, 27, 18] // all operations done mod p
```

---

## Naive Baseline

```cpp
// p = 18446744069414584321ULL
static __host__ __device__ inline uint64_t add_mod(uint64_t a, uint64_t b, uint64_t p){
    uint64_t s = a + b; return (s < a || s >= p) ? s - p : s;
}
static __host__ __device__ inline uint64_t mul_mod(uint64_t a, uint64_t b, uint64_t p){
    unsigned __int128 prod = (unsigned __int128)a * (unsigned __int128)b;
    return (uint64_t)(prod % p);
}

for (size_t i = 0; i < n; ++i)
  for (size_t j = 0; j < n; ++j)
    d_output[i + j] = add_mod(d_output[i + j], mul_mod(d_input1[i], d_input2[j], p), p);
```

---

### Optimizations to consider

- NTT-based multiplication over Goldilocks (supports large power-of-two sizes).

- Precompute and reuse twiddle factors in constant/texture memory.

- Use Montgomery or Barrett reduction for mul_mod.

- Use shared memory tiling and register blocking.
