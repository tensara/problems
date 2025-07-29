---
slug: "shortest-path"
title: "Single Source Shortest Path"
difficulty: "MEDIUM"
author: "assistant"
tags: ["graph", "shortest-path", "dijkstra"]
parameters:
  - name: "d_adj_matrix"
    type: "[VAR]"
    pointer: "true"
    const: "true"
  
  - name: "source"
    type: "int"
    pointer: "false"
    constant: "false"
  
  - name: "d_distances"
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "n" 
    type: "size_t"
    pointer: "false"
    constant: "false"
    
---

Find the shortest path distances from a source node to all other nodes in a weighted directed graph.

Given a weighted adjacency matrix $A$ of size $N \times N$ with integer weights and a source node $s$, compute the shortest distances from $s$ to all nodes:

$$
d[v] = \min_{path\ from\ s\ to\ v} \sum_{(u,w) \in path} A[u][w]
$$

## Input
- Weighted adjacency matrix $A$ of size $N \times N$ where $A[i][j]$ contains the integer weight of the edge from node $i$ to node $j$, and $A[i][j] = 0$ if no edge exists
- Source node index $s$

## Output
- Array $d$ of size $N$ containing shortest distances from source $s$ to all nodes. If a node is unreachable, its distance should be $\infty$.

