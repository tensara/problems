---
slug: "n-body-simulation"
title: "N-Body Gravitational Simulation"
difficulty: "HARD"
author: "sarthak"
tags: ["physics", "simulation", "numerical"]
parameters:
  - name: "positions"
    type: "[VAR]"
    pointer: "true"
    const: "true"
  
  - name: "velocities"
    type: "[VAR]"
    pointer: "true"
    const: "true"
    
  - name: "masses"
    type: "[VAR]"
    pointer: "true"
    const: "true"
    
  - name: "new_positions"
    type: "[VAR]"
    pointer: "true"
    const: "false"
    
  - name: "n_bodies"
    type: "size_t"
    pointer: "false"
    constant: "false"
    
  - name: "dt"
    type: "float"
    pointer: "false"
    constant: "false"
    
  - name: "G"
    type: "float"
    pointer: "false"
    constant: "false"

---

Simulate gravitational interactions between N bodies using Newton's law of universal gravitation. Compute the new positions of all bodies after one timestep.

## Physics:

**Gravitational Force** between two bodies with masses $m_1$ and $m_2$ separated by distance $r$:
$$\vec{F} = G \frac{m_1 m_2}{r^2} \hat{r}$$

Where:
- $G$ is the gravitational constant
- $\hat{r}$ is the unit vector from body 1 to body 2
- $r$ is the distance between the bodies

**Newton's Second Law**: $\vec{F} = m \vec{a}$, so acceleration is:
$$\vec{a} = \frac{\vec{F}}{m}$$

**Numerical Integration** using Euler's method:
- $\vec{v}_{new} = \vec{v}_{old} + \vec{a} \cdot dt$
- $\vec{x}_{new} = \vec{x}_{old} + \vec{v}_{new} \cdot dt$

## Algorithm:

1. **Force Calculation**: For each body $i$, calculate the total gravitational force from all other bodies:
   $$\vec{F}_i = \sum_{j \neq i} G \frac{m_i m_j}{|\vec{r}_{ij}|^2} \frac{\vec{r}_{ij}}{|\vec{r}_{ij}|}$$

2. **Velocity Update**: Update velocities using force and mass:
   $$\vec{v}_{i,new} = \vec{v}_{i,old} + \frac{\vec{F}_i}{m_i} \cdot dt$$

3. **Position Update**: Update positions using new velocities:
   $$\vec{x}_{i,new} = \vec{x}_{i,old} + \vec{v}_{i,new} \cdot dt$$

## Input:
- Current positions: $N \times 3$ array (x, y, z coordinates)
- Current velocities: $N \times 3$ array (vx, vy, vz components)
- Masses: $N \times 1$ array (mass of each body)
- Number of bodies: $N$
- Time step: $dt$
- Gravitational constant: $G$

## Output:
- New positions: $N \times 3$ array (updated x, y, z coordinates)

## Implementation Notes:
- **Softening Parameter**: Add small epsilon ($\epsilon = 10^{-3}$) to distance to avoid division by zero
- **Symmetry**: Force between bodies $i$ and $j$ is equal and opposite (Newton's third law)
- **Complexity**: $O(N^2)$ for naive implementation due to all-pairs force calculation
- **Numerical Stability**: Use appropriate time step to maintain stability

## GPU Optimization:
- **Parallelization**: Each thread can compute forces for one body
- **Shared Memory**: Cache positions and masses in shared memory for repeated access
- **Reduction**: Use parallel reduction for force accumulation
- **Memory Coalescing**: Ensure coalesced access to position, velocity, and mass arrays

## Advanced Techniques:
- **Barnes-Hut Algorithm**: Use octree to approximate distant forces ($O(N \log N)$)
- **Fast Multipole Method**: Further optimize to $O(N)$ complexity
- **Verlet Integration**: Use more stable integration scheme
- **Adaptive Time Stepping**: Adjust $dt$ based on system dynamics

## Physical Applications:
- Galaxy formation and evolution
- Planetary system dynamics
- Stellar cluster simulations
- Asteroid and comet trajectory prediction 