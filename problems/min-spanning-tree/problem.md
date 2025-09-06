---
slug: "min-spanning-tree"
title: "Minimum Spanning Tree"
difficulty: "MEDIUM"
author: "sarthak"
tags: ["graphs"]
parameters:
  - name: "A"
    type: "[VAR]"
    pointer: "true"
    const: "true"
  
  - name: "min_weight"
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "n" 
    type: "size_t"
    pointer: "false"
    constant: "false"

---

Find the minimum spanning tree of a weighted undirected graph.

Given a weighted adjacency matrix $A$ of size $n \times n$ with positive integer weights, find the minimum spanning tree (MST) that connects all vertices with minimum total edge weight:

$$
MST = \arg\min_{T \in \text{spanning trees}} \sum_{(u,v) \in T} A[u][v]
$$

The MST connects all $n$ vertices using exactly $n - 1$ edges with minimum total weight.

## Input
- Weighted adjacency matrix $A$ of size $n \times n$ where $A[i][j]$ contains the positive integer weight of the edge between nodes $i$ and $j$, and $A[i][j] = 0$ if no edge exists
- The matrix is symmetric: $A[i][j] = A[j][i]$ (undirected graph)

## Output
- Total weight of the minimum spanning tree, written to `min_weight`. Return $-1$ if graph is not connected.