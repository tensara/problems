---
slug: "k-means-clustering"
title: "K-means Clustering"
difficulty: "HARD"
author: "sarthak"
tags: ["machine-learning", "clustering", "optimization"]
parameters:
  - name: "points"
    type: "[VAR]"
    pointer: "true"
    const: "true"
  
  - name: "centroids"
    type: "[VAR]"
    pointer: "true"
    const: "false"
    
  - name: "n_points"
    type: "size_t"
    pointer: "false"
    constant: "false"
    
  - name: "n_features"
    type: "size_t"
    pointer: "false"
    constant: "false"
    
  - name: "k"
    type: "size_t"
    pointer: "false"
    constant: "false"
    
  - name: "max_iter"
    type: "int"
    pointer: "false"
    constant: "false"

---

Implement the K-means clustering algorithm to partition n data points into k clusters. The algorithm iteratively assigns points to their nearest centroid and updates centroids until convergence.

## Algorithm:

1. **Initialize**: Start with k initial centroids (given as input)
2. **Assignment Step**: For each point, find the nearest centroid using Euclidean distance
3. **Update Step**: Recalculate each centroid as the mean of all points assigned to it
4. **Convergence Check**: Repeat steps 2-3 until centroids stop changing or max iterations reached

## Mathematical Formulation:

**Distance Calculation**: For point $\mathbf{x}_i$ and centroid $\mathbf{c}_j$:
$$d(\mathbf{x}_i, \mathbf{c}_j) = \sqrt{\sum_{d=1}^{D} (x_{i,d} - c_{j,d})^2}$$

**Assignment**: Assign point $\mathbf{x}_i$ to cluster $j^*$ where:
$$j^* = \arg\min_{j} d(\mathbf{x}_i, \mathbf{c}_j)$$

**Centroid Update**: Update centroid $\mathbf{c}_j$ as:
$$\mathbf{c}_j = \frac{1}{|S_j|} \sum_{\mathbf{x}_i \in S_j} \mathbf{x}_i$$
where $S_j$ is the set of points assigned to cluster $j$.

## Input:
- Data points matrix: $n \times d$ (n_points × n_features)
- Initial centroids matrix: $k \times d$ (k × n_features)
- Number of points: $n$
- Number of features: $d$
- Number of clusters: $k$
- Maximum iterations: `max_iter`

## Output:
- Final centroids matrix: $k \times d$ (k × n_features)
- Centroids are updated in-place in the input array

## Implementation Notes:
- Handle empty clusters by keeping the previous centroid
- Use early termination when centroids converge (change < threshold)
- Euclidean distance can be computed without square root for comparison
- Algorithm may converge to different local minima depending on initialization

## GPU Optimization Strategies:
- **Distance Calculation**: Each point-centroid distance can be computed independently
- **Assignment**: Use reduction operations to find minimum distances
- **Centroid Update**: Use segmented reduction to compute cluster means
- **Memory Access**: Ensure coalesced memory access patterns for points and centroids
- **Shared Memory**: Cache centroids in shared memory for repeated distance calculations

## Parallelization Approaches:
1. **Point-Parallel**: Each thread handles one point's assignment
2. **Cluster-Parallel**: Each thread block handles one cluster's centroid update
3. **Hybrid**: Combine both approaches for different algorithm phases

## Convergence:
The algorithm terminates when centroids change by less than a threshold or when maximum iterations are reached. Multiple runs with different initializations can help find better solutions. 