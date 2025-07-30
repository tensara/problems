# Graph Coloring

## Problem Statement

Given an undirected graph represented as an adjacency matrix, assign colors to vertices such that no two adjacent vertices have the same color, using the minimum number of colors possible with a greedy algorithm.

## Input

- `adj_matrix`: A 2D tensor of shape `(N, N)` representing the undirected adjacency matrix
  - `adj_matrix[i][j]` = 1 if there is an edge between vertices `i` and `j`
  - `adj_matrix[i][j]` = 0 if there is no edge
  - The matrix is symmetric: `adj_matrix[i][j] = adj_matrix[j][i]`
  - Diagonal elements are 0 (no self-loops)

## Output

- A 1D tensor of shape `(N,)` containing the color assignment for each vertex
- `output[i]` = color assigned to vertex `i` (starting from color 0)
- Adjacent vertices must have different colors

## Algorithm Details

The greedy coloring algorithm processes vertices in order and assigns the smallest available color:

1. Initialize all vertices as uncolored
2. For each vertex v from 0 to N-1:
   - Find all neighbors of v that are already colored
   - Collect the colors used by these neighbors
   - Assign the smallest non-negative integer color not used by any neighbor

## Complexity

- Time Complexity: O(N + E) where E is the number of edges
- Space Complexity: O(N)

## Example

Input:
```
adj_matrix = [
    [0, 1, 0, 0, 1],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 1],
    [0, 0, 1, 0, 1],
    [1, 0, 1, 1, 0]
]
```

This represents a pentagon graph with one diagonal edge (2-4).

Output:
```
colors = [0, 1, 0, 1, 2]
```

Verification:
- Vertex 0 (color 0) is adjacent to vertices 1 (color 1) and 4 (color 2) ✓
- Vertex 1 (color 1) is adjacent to vertices 0 (color 0) and 2 (color 0) ✓  
- Vertex 2 (color 0) is adjacent to vertices 1 (color 1), 3 (color 1), and 4 (color 2) ✓
- Vertex 3 (color 1) is adjacent to vertices 2 (color 0) and 4 (color 2) ✓
- Vertex 4 (color 2) is adjacent to vertices 0 (color 0), 2 (color 0), and 3 (color 1) ✓

## Implementation Notes

- The greedy algorithm doesn't always produce optimal colorings but is efficient
- Vertex ordering can affect the number of colors used
- For dense graphs with cliques, more colors will be required
- Handle edge cases like empty graphs or isolated vertices
- Use efficient data structures for neighbor lookup and color checking 