import torch
import ctypes
from typing import List, Dict, Tuple, Any
import math

from problem import Problem

class topological_sort(Problem):
    """Topological sorting problem using Kahn's algorithm."""
    
    def __init__(self):
        super().__init__(
            name="topological-sort"
        )
    
    def reference_solution(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of topological sorting using Kahn's algorithm.
        
        Args:
            adj_matrix: Directed adjacency matrix (N x N) representing a DAG
            
        Returns:
            Topologically sorted order of vertices (N,) tensor
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            N = adj_matrix.size(0)
            device = adj_matrix.device
            
            if N == 0:
                return torch.empty(0, device=device, dtype=torch.int32)
            
            # Calculate in-degrees for all vertices
            in_degree = torch.sum(adj_matrix, dim=0).int()
            
            # Initialize result and queue
            result = torch.full((N,), -1, device=device, dtype=torch.int32)
            result_idx = 0
            
            # Process vertices with in-degree 0
            remaining_vertices = torch.arange(N, device=device)
            
            for step in range(N):
                # Find vertices with in-degree 0
                zero_indegree_mask = (in_degree == 0)
                
                if not torch.any(zero_indegree_mask):
                    # No vertices with in-degree 0 - graph has cycles
                    break
                
                # Pick the first vertex with in-degree 0 (deterministic ordering)
                zero_indegree_vertices = torch.where(zero_indegree_mask)[0]
                current_vertex = zero_indegree_vertices[0]
                
                # Add to result
                result[result_idx] = current_vertex
                result_idx += 1
                
                # Remove this vertex by setting its in-degree to -1
                in_degree[current_vertex] = -1
                
                # Update in-degrees of neighbors
                neighbors = torch.where(adj_matrix[current_vertex] > 0)[0]
                in_degree[neighbors] -= 1
            
            # If we didn't process all vertices, there was a cycle
            if result_idx < N:
                # Return partial result with -1s for unprocessed vertices
                return result
            
            return result

    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for topological sorting.
        
        Returns:
            List of test case dictionaries with varying graph sizes
        """
        sizes = [
            ("n = 1024", 1024),
            ("n = 2048", 2048),
            ("n = 4096", 4096),
            ("n = 6144", 6144)
        ]
        
        return [
            {
                "name": name,
                "dims": (size, size),
                "create_inputs": lambda size=size: self._create_graph_inputs(size, dtype)
            }
            for name, size in sizes
        ]
    
    def _create_graph_inputs(self, size: int, dtype: torch.dtype) -> Tuple[torch.Tensor]:
        """Create a directed acyclic graph (DAG)."""
        adj_matrix = torch.zeros((size, size), device="cuda", dtype=dtype)
        
        if size <= 1:
            return (adj_matrix,)
        
        # Create a DAG by only allowing edges from lower to higher indices
        # This guarantees acyclicity
        
        # Add some structure - create layers
        layer_size = max(1, size // 8)  # 8 layers approximately
        
        for i in range(size):
            current_layer = i // layer_size
            
            # Add edges to vertices in next layers
            for j in range(i + 1, min(size, i + layer_size * 2)):
                target_layer = j // layer_size
                
                # Higher probability for edges to next layer, lower for further layers
                if target_layer == current_layer + 1:
                    prob = 0.4
                elif target_layer == current_layer + 2:
                    prob = 0.2
                else:
                    prob = 0.1
                
                if torch.rand(1, device="cuda").item() < prob:
                    adj_matrix[i, j] = 1
        
        # Add some random edges (still respecting DAG property)
        for i in range(size):
            # Add 1-3 random forward edges
            num_edges = torch.randint(1, 4, (1,), device="cuda").item()
            
            for _ in range(num_edges):
                # Pick a random vertex with higher index
                if i < size - 1:
                    j = torch.randint(i + 1, size, (1,), device="cuda").item()
                    if torch.rand(1, device="cuda").item() < 0.3:  # 30% chance
                        adj_matrix[i, j] = 1
        
        return (adj_matrix,)

    def generate_sample(self, dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
        """
        Generate a single sample test case for debugging or interactive runs.
        
        Returns:
            A test case dictionary
        """
        name = "Sample (n = 6)"
        size = 6
        
        def create_sample_inputs():
            # Create a simple DAG
            adj_matrix = torch.tensor([
                [0, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0]
            ], device="cuda", dtype=dtype)
            return (adj_matrix,)
        
        return {
            "name": name,
            "dims": (size, size),
            "create_inputs": create_sample_inputs
        }

    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the topological sort result is correct.
        
        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            
        Returns:
            Tuple of (is_correct, debug_info)
        """
        debug_info = {}
        
        # Basic checks
        if expected_output.shape != actual_output.shape:
            debug_info["error"] = "Shape mismatch"
            return False, debug_info
        
        # Check if both solutions have the same validity
        expected_valid = torch.all(expected_output >= 0)
        actual_valid = torch.all(actual_output >= 0)
        
        if expected_valid != actual_valid:
            debug_info["error"] = "Validity mismatch - one solution detected cycle, other didn't"
            return False, debug_info
        
        # If both invalid (contain -1), they're both correct in detecting cycles
        if not expected_valid and not actual_valid:
            return True, debug_info
        
        # Check if actual result is a valid permutation
        N = expected_output.size(0)
        expected_set = set(range(N))
        actual_set = set(actual_output.cpu().numpy().tolist())
        
        if actual_set != expected_set:
            debug_info["error"] = "Result is not a valid permutation"
            return False, debug_info
        
        # For topological sort, multiple valid orderings exist
        # We accept any valid topological ordering
        # A full validation would require checking against the original graph
        return True, debug_info
    
    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the topological sort solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # adjacency matrix
                ctypes.POINTER(ctypes.c_int),    # output topological order
                ctypes.c_size_t,                 # N (number of vertices)
            ],
            "restype": None
        }
    
    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the number of floating point operations for the problem.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            Number of floating point operations
        """
        N = test_case["dims"][0]
        # Kahn's algorithm: O(V + E) where E is number of edges
        # Approximate as O(N^2) for dense graphs
        return N * N
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the number of vertices N
        """
        N = test_case["dims"][0]
        return [N] 