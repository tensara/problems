---
slug: "ecc-point-negation"
title: "ECC Point Negation (Batched)"
difficulty: "EASY"
author: "tensara"
tags: ["crypto"]
parameters:
  - name: "xs"
    type: "uint64_t"
    pointer: "true"
    const: "true"

  - name: "ys"
    type: "uint64_t"
    pointer: "true"
    const: "true"

  - name: "p"
    type: "uint64_t"
    pointer: "false"
    constant: "true"

  - name: "out_xy"
    type: "uint64_t"
    pointer: "true"
    const: "false"

  - name: "n"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Negate **N** elliptic curve points in parallel over the Tensara curve:

$$
E: y^2 \equiv x^3 + 7 \pmod{p}, \quad p = 2^{61} - 1.
$$

For each input point \($x_i, y_i$\):

$$
(x_i, y_i) \mapsto (x_i,\; (p - y_i) \bmod p).
$$

## Input

- Arrays `xs[i]`, `ys[i]` of length \(N\), each element in \([0, p)\).
- Prime modulus $p = 2^{61} - 1$.
- Batch size \(N\).

## Output

- A single array `out_xy` of length \(2N\), storing the results as pairs:
  - `out_xy[2*i] = xs[i]`
  - `out_xy[2*i + 1] = (p - ys[i]) % p`

## Correctness

Your kernel must produce the exact negation for every point in the batch.
