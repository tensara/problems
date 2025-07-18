---
slug: "monte-carlo-pi"
title: "Monte Carlo Pi Estimation"
difficulty: "MEDIUM"
author: "sarthak"
tags: ["numerical", "monte-carlo", "parallel"]
parameters:
  - name: "random_x"
    type: "[VAR]"
    pointer: "true"
    const: "true"
  
  - name: "random_y" 
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "pi_estimate"
    type: "[VAR]"
    pointer: "true"
    const: "false"
    
  - name: "n_samples" 
    type: "size_t"
    pointer: "false"
    constant: "false"

---

Estimate the value of π using the Monte Carlo method. This classic numerical method uses random sampling to approximate π by simulating random points in a unit square and counting how many fall inside a unit circle.

## Algorithm:

1. Generate random points $(x, y)$ uniformly distributed in the unit square $[0, 1] \times [0, 1]$
2. For each point, check if it lies inside the unit circle: $x^2 + y^2 \leq 1$
3. Count the number of points inside the circle
4. Estimate π using the ratio: $\pi \approx 4 \times \frac{\text{points inside circle}}{\text{total points}}$

## Mathematical Foundation:

The area of a unit circle is $\pi r^2 = \pi \cdot 1^2 = \pi$.
The area of the unit square is $1 \times 1 = 1$.

The ratio of the circle's area to the square's area is $\frac{\pi}{4}$.

Therefore:
$$
\frac{\text{points inside circle}}{\text{total points}} \approx \frac{\pi}{4}
$$

Solving for π:
$$
\pi \approx 4 \times \frac{\text{points inside circle}}{\text{total points}}
$$

## Input:
- Array of random x-coordinates in $[0, 1]$
- Array of random y-coordinates in $[0, 1]$
- Number of samples $n$

## Output:
- Estimated value of π (single float)

## Implementation Notes:
- Each sample point can be processed independently (embarrassingly parallel)
- Use atomic operations or reduction to count points inside the circle
- Higher sample counts lead to more accurate estimates
- Expected accuracy improves roughly as $\frac{1}{\sqrt{n}}$

## GPU Optimization:
- Each thread can process one or more sample points
- Use shared memory for local counting before global reduction
- Coalesced memory access for reading random coordinates
- Consider using cuRAND for on-device random number generation 