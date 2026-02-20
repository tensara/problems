---
slug: "ecc-point-negation"
title: "ECC Point Negation (Batched)"
difficulty: "EASY"
author: "soham"
tags: ["crypto"]
---

Negate **N** elliptic curve points in parallel over the curve:

$$
E: y^2 \equiv x^3 + 7 \pmod{p}, \quad p = 2^{61} - 1.
$$

For each input point $(x_i, y_i)$:

$$
(x_i, y_i) \mapsto (x_i,\; (p - y_i) \bmod p).
$$


## Input

- Arrays `xs[i]`, `ys[i]` of length $N$, each element in $[0, p)$
- Prime modulus $p = 2^{61} - 1$
- Batch size $N$

Your kernel must produce the exact negation for every point in the batch.

## Output

- A single array `out_xy` of length $2N$, storing the results as pairs:
  - `out_xy[2*i] = xs[i]`
  - `out_xy[2*i + 1] = (p - ys[i]) % p`
