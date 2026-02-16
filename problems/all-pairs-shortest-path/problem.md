---
slug: "all-pairs-shortest-path"
title: "All-Pairs Shortest Path"
difficulty: "MEDIUM"
author: "sarthak"
tags: ["graphs"]
parameters:
  - name: "adj_matrix"
    type: "float"
    pointer: "true"
    const: "true"
  
  - name: "output"
    type: "float"
    pointer: "true"
    const: "false"

  - name: "n" 
    type: "size_t"
    pointer: "false"
    constant: "false"
    
---

Given a weighted directed graph represented as an adjacency matrix, compute the shortest distances between all pairs of vertices using the Floyd-Warshall algorithm.

## Input

- `adj_matrix`: A 2D tensor of shape $(n, n)$ representing the weighted adjacency matrix
  - `adj_matrix[i][j]` = weight of edge from vertex `i` to vertex `j`
  - `adj_matrix[i][j]` = 0 if there is no direct edge (except diagonal which represents self-loops)
  - All weights are positive integers

## Output

- A 2D tensor of shape $(n, n)$ containing the shortest distances between all pairs of vertices
- `output[i][j]` = shortest distance from vertex `i` to vertex `j`
- If no path exists between vertices `i` and `j`, the distance should be `-1`
