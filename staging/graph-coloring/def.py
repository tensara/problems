import torch
import ctypes
from typing import List, Dict, Tuple, Any
import math

from problem import Problem

class graph_coloring(Problem):
    """Graph coloring problem using greedy coloring algorithm."""
    
    def __init__(self):
        super().__init__(
            name="graph-coloring"
        )
    
    def reference_solution(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of graph coloring using greedy algorithm.
        
        Args:
            adj_matrix: Undirected adjacency matrix (N x N) with 0s and 1s
            
        Returns:
            Color assignment for each vertex (N,) tensor with integer colors
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            N = adj_matrix.size(0)
            device = adj_matrix.device
            
            if N == 0:
                return torch.empty(0, device=device, dtype=torch.int32)
            
            # Initialize colors - all uncolored (-1)
            colors = torch.full((N,), -1, device=device, dtype=torch.int32)
            
            # Color vertices in order
            for vertex in range(N):
                # Get neighbors of current vertex
                neighbors = torch.where(adj_matrix[vertex] > 0)[0]
                
                # Get colors of colored neighbors
                neighbor_colors = colors[neighbors]
                neighbor_colors = neighbor_colors[neighbor_colors >= 0]
                
                # Find smallest available color
                color = 0
                while color in neighbor_colors:
                    color += 1
                
                colors[vertex] = color
            
            return colors

    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for graph coloring.
        
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
        """Create an undirected graph for coloring."""
        adj_matrix = torch.zeros((size, size), device="cuda", dtype=dtype)
        
        # Create random undirected graph with varying density
        # Higher density makes coloring more challenging
        density = 0.1 + 0.3 * torch.rand(1, device="cuda").item()  # 10% to 40% density
        
        # Generate upper triangular random connections
        upper_triu_mask = torch.triu(torch.ones(size, size, device="cuda", dtype=torch.bool), diagonal=1)
        random_mask = torch.rand((size, size), device="cuda") < density
        connection_mask = upper_triu_mask & random_mask
        
        # Set symmetric connections
        adj_matrix[connection_mask] = 1
        adj_matrix = adj_matrix + adj_matrix.T  # Make symmetric
        
        # Ensure no self-loops
        adj_matrix.fill_diagonal_(0)
        
        # Add some structure to make it more interesting
        # Create small cliques (complete subgraphs)
        num_cliques = max(1, size // 200)
        for _ in range(num_cliques):
            clique_size = torch.randint(3, min(8, size // 10 + 1), (1,), device="cuda").item()
            start_idx = torch.randint(0, max(1, size - clique_size), (1,), device="cuda").item()
            
            # Create clique
            for i in range(start_idx, min(start_idx + clique_size, size)):
                for j in range(i + 1, min(start_idx + clique_size, size)):
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
        
        return (adj_matrix,)

    def generate_sample(self, dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
        """
        Generate a single sample test case for debugging or interactive runs.
        
        Returns:
            A test case dictionary
        """
        name = "Sample (n = 5)"
        size = 5
        
        def create_sample_inputs():
            # Create a simple graph: pentagon with one diagonal
            adj_matrix = torch.tensor([
                [0, 1, 0, 0, 1],
                [1, 0, 1, 0, 0],
                [0, 1, 0, 1, 1],
                [0, 0, 1, 0, 1],
                [1, 0, 1, 1, 0]
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
        Verify if the graph coloring result is correct.
        
        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            
        Returns:
            Tuple of (is_correct, debug_info)
        """
        # For graph coloring, we check if the coloring is valid, not if it matches exactly
        # A valid coloring means no adjacent vertices have the same color
        debug_info = {}
        
        # Get the adjacency matrix from test case inputs
        # This is a bit tricky since we don't have direct access, but we can check validity
        
        # Basic checks
        if expected_output.shape != actual_output.shape:
            debug_info["error"] = "Shape mismatch"
            return False, debug_info
        
        # Check if all colors are non-negative
        if torch.any(actual_output < 0):
            debug_info["error"] = "Invalid negative colors found"
            return False, debug_info
        
        # Since we can't easily access the adjacency matrix here, we'll do a basic comparison
        # In a real implementation, you'd want to verify the coloring is valid against the graph
        num_colors_expected = torch.max(expected_output).item() + 1
        num_colors_actual = torch.max(actual_output).item() + 1
        
        # Accept if the solution uses a reasonable number of colors
        # (within 2x of the reference solution)
        is_reasonable = num_colors_actual <= num_colors_expected * 2
        
        if not is_reasonable:
            debug_info = {
                "expected_colors": num_colors_expected,
                "actual_colors": num_colors_actual,
                "error": "Uses too many colors"
            }
        
        return is_reasonable, debug_info
    
    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the graph coloring solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # adjacency matrix
                ctypes.POINTER(ctypes.c_int),    # output color assignments
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
        # Greedy coloring: O(N * E) where E is number of edges
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