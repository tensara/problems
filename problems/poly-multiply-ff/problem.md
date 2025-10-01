---
slug: "poly-multiply-ff"
title: "Polynomial Multiplication over Finite Field "
difficulty: "MEDIUM"
author: "soham"
tags: ["crypto"]
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

Multiply two polynomials over the Mersenne field:

$$
p = 2^{31} - 1 = 2147483647
$$

Let $a(x)$ and $b(x)$ be polynomials of degree $n-1$ with coefficients in $[0, p)$:

$$
a(x) = \sum_{i=0}^{n-1} a_i x^i, \qquad
b(x) = \sum_{j=0}^{n-1} b_j x^j
$$

Compute their product modulo $p$:

$$
c(x) = a(x) \cdot b(x) \pmod{p}
$$

The result is a coefficient vector of length $2n - 1$:

$$
c_k = \sum_{i+j=k} a_i \, b_j \pmod{p}
$$

## Input

- Two arrays `d_input1`, `d_input2` of length $n$, with values in $[0, p)$.
- $n$ is a power of two (e.g., 64, 256, 1024).

## Output

- Array `d_output` of length $2n - 1$ with coefficients of $c(x)$ modulo $p$.

## Notes

For small $n=3$:

$$
a(x) = 1 + 2x + 3x^2, \qquad
b(x) = 4 + 5x + 6x^2
$$

Their product is:

$$
c(x) = 4 + 13x + 28x^2 + 27x^3 + 18x^4 \pmod{p}
$$

So:

- $a = [1, 2, 3]$
- $b = [4, 5, 6]$
- $c = [4, 13, 28, 27, 18]$

A naive implementation:
```cpp
const uint32_t P = 2147483647u;

static __host__ __device__ inline uint32_t add_mod(uint32_t a, uint32_t b) {
    uint64_t s = (uint64_t)a + b;
    return (uint32_t)(s % P);
}

static __host__ __device__ inline uint32_t mul_mod(uint32_t a, uint32_t b) {
    uint64_t prod = (uint64_t)a * b;
    return (uint32_t)(prod % P);
}

for (size_t i = 0; i < n; ++i) {
  for (size_t j = 0; j < n; ++j) {
    d_output[i + j] = add_mod(d_output[i + j], mul_mod(d_input1[i], d_input2[j]));
  }
}
```