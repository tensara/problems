# All-Pairs Shortest Path

## Problem Statement

Given a weighted directed graph represented as an adjacency matrix, compute the shortest distances between all pairs of vertices using the Floyd-Warshall algorithm.

## Input

- `adj_matrix`: A 2D tensor of shape `(N, N)` representing the weighted adjacency matrix
  - `adj_matrix[i][j]` = weight of edge from vertex `i` to vertex `j`
  - `adj_matrix[i][j]` = 0 if there is no direct edge (except diagonal which represents self-loops)
  - All weights are positive integers

## Output

- A 2D tensor of shape `(N, N)` containing the shortest distances between all pairs of vertices
- `output[i][j]` = shortest distance from vertex `i` to vertex `j`
- If no path exists between vertices `i` and `j`, the distance should be infinity

## Algorithm Details

The Floyd-Warshall algorithm works by considering all vertices as intermediate points:

1. Initialize the distance matrix with the input adjacency matrix
2. Set distances from unreachable vertices (where adj_matrix[i][j] = 0 and i ≠ j) to infinity
3. Set diagonal elements to 0 (distance from vertex to itself)
4. For each vertex k from 0 to N-1:
   - For each pair of vertices (i, j):
     - Update distance[i][j] = min(distance[i][j], distance[i][k] + distance[k][j])

## Complexity

- Time Complexity: O(N³)
- Space Complexity: O(N²)

## Example

Input:
```
adj_matrix = [
    [0, 3, 8, ∞],
    [∞, 0, ∞, 1],
    [∞, 4, 0, ∞],
    [2, ∞, ∞, 0]
]
```

Output:
```
distances = [
    [0, 3, 7, 4],
    [3, 0, 4, 1],
    [7, 4, 0, 5],
    [2, 5, 9, 0]
]
```

## Implementation Notes

- The algorithm should handle disconnected components gracefully
- Use GPU-optimized operations for large graphs
- Leverage broadcasting for efficient matrix operations
- Handle numerical precision carefully when dealing with infinity values 