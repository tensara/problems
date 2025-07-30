# Topological Sort

## Problem Statement

Given a directed acyclic graph (DAG) represented as an adjacency matrix, find a topological ordering of its vertices using Kahn's algorithm. A topological ordering is a linear ordering of vertices such that for every directed edge (u, v), vertex u comes before vertex v in the ordering.

## Input

- `adj_matrix`: A 2D tensor of shape `(N, N)` representing the directed adjacency matrix
  - `adj_matrix[i][j]` = 1 if there is a directed edge from vertex `i` to vertex `j`
  - `adj_matrix[i][j]` = 0 if there is no edge
  - The graph must be acyclic (DAG) for a valid topological ordering to exist

## Output

- A 1D tensor of shape `(N,)` containing the topologically sorted vertices
- `output[i]` = the i-th vertex in topological order
- If the graph contains cycles, some elements may be -1 to indicate invalid ordering

## Algorithm Details

Kahn's algorithm works by repeatedly removing vertices with no incoming edges:

1. Calculate the in-degree (number of incoming edges) for each vertex
2. Initialize a queue with all vertices that have in-degree 0
3. While the queue is not empty:
   - Remove a vertex v from the queue and add it to the result
   - For each neighbor u of v:
     - Decrease the in-degree of u by 1
     - If in-degree of u becomes 0, add u to the queue
4. If all vertices are processed, return the topological ordering
5. If some vertices remain unprocessed, the graph contains cycles

## Complexity

- Time Complexity: O(V + E) where V is vertices and E is edges
- Space Complexity: O(V)

## Example

Input:
```
adj_matrix = [
    [0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0]
]
```

This represents a DAG with edges: 0→1, 0→2, 1→3, 1→4, 2→4, 2→5, 3→5, 4→5

Initial in-degrees: [0, 1, 1, 1, 2, 3]

Output:
```
topological_order = [0, 1, 2, 3, 4, 5]
```

Step-by-step execution:
1. Start with vertex 0 (in-degree 0)
2. Remove 0, update in-degrees: [-, 0, 0, 1, 2, 3]
3. Add vertices 1 and 2 to queue
4. Process vertex 1, update in-degrees: [-, -, 0, 0, 1, 3]
5. Add vertices 3 and 4 to queue (4's in-degree becomes 1)
6. Continue until all vertices are processed

## Implementation Notes

- Handle cycles gracefully by detecting when no vertices have in-degree 0
- Use efficient data structures for queue operations
- Multiple valid topological orderings may exist for the same DAG
- Ensure deterministic output by processing vertices in a consistent order
- For performance, leverage GPU parallelism where possible
- Consider memory access patterns for large graphs 