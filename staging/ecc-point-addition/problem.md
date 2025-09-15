---
slug: "ecc-point-addition"
title: "ECC Point Addition (Batched)"
difficulty: "EASY"
author: "tensara"
tags: ["crypto"]
parameters:
  - name: "xs1"
    type: "uint64_t"
    pointer: "true"
    const: "true"

  - name: "ys1"
    type: "uint64_t"
    pointer: "true"
    const: "true"

  - name: "xs2"
    type: "uint64_t"
    pointer: "true"
    const: "true"

  - name: "ys2"
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

Add **N** elliptic-curve point pairs in parallel over the Tensara curve:

\[
E:\ y^2 \equiv x^3 + 7 \pmod p,\quad p=2^{61}-1.
\]

For each pair \((x_1,y_1)\) and \((x_2,y_2)\), compute \(R=(x_3,y_3)=(x_1,y_1)+(x_2,y_2)\) with the usual group law over \(\mathbb{F}\_p\).

## Input

- Arrays `xs1[i]`, `ys1[i]`, `xs2[i]`, `ys2[i]` of length \(N\), each element in \([0,p)\).
- Prime modulus \(p = 2^{61}-1\).
- Batch size \(N\).

## Output

- A single array `out_xy` of length \(2N\), storing results as pairs:
  - `out_xy[2*i] = x3`
  - `out_xy[2*i + 1] = y3`

## Group Law (over \(\mathbb{F}\_p\))

We assume inputs do **not** produce the point at infinity.  
That means denominators in the slope formulas are always nonzero.

For \(P=(x_1,y_1)\), \(Q=(x_2,y_2)\):

- If \(P \neq Q\):
  \[
  \lambda = (y_2 - y_1)\,(x_2 - x_1)^{-1} \pmod p
  \]

- If \(P = Q\) (doubling):
  \[
  \lambda = \frac{3x_1^2}{2y_1} \pmod p
  \]

Then
\[
\begin{aligned}
x_3 &= \lambda^2 - x_1 - x_2 \pmod p,\\
y_3 &= \lambda\,(x_1 - x_3) - y_1 \pmod p.
\end{aligned}
\]

## Correctness

Your kernel must:

- Perform all arithmetic modulo \(p\).
- Use modular inverses where required.
- Return exactly the same result as the reference.

## Suggested Batches

- \(N \in \{262{,}144,\ 524{,}288,\ 1{,}048{,}576,\ 2{,}097{,}152\}\)

## Performance Metric

This task is integer-heavy with modular inverses.  
We’ll count ~12 ops per element, but score mainly by runtime.
