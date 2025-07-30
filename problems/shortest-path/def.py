import torch
import ctypes
from typing import List, Dict, Tuple, Any
import math

from problem import Problem

class shortest_path(Problem):
    """Single source shortest path problem using Dijkstra's algorithm."""
    
    def __init__(self):
        super().__init__(
            name="shortest-path"
        )
    
    def reference_solution(self, adj_matrix: torch.Tensor, source: int) -> torch.Tensor:
        """
        PyTorch implementation of single source shortest path.
        
        Args:
            adj_matrix: Weighted adjacency matrix (N x N) with integer weights
            source: Source node index
            
        Returns:
            Shortest distances from source to all nodes
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            N = adj_matrix.size(0)
            device = adj_matrix.device

            dist = torch.full((N,), float('inf'), device=device)
            dist[source] = 0.0

            u, v = torch.where(adj_matrix > 0)
            weights = adj_matrix[u, v]

            for i in range(N - 1):
                new_dist = dist[u] + weights
                dist.scatter_reduce_(0, v, new_dist, reduce='amin')

            return dist

    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for shortest path.
        
        Returns:
            List of test case dictionaries with varying graph sizes
        """
        sizes = [
            ("n = 2048", 2048),
            ("n = 4096", 4096),
            ("n = 6144", 6144),
            ("n = 8192", 8192)
        ]
        
        return [
            {
                "name": name,
                "dims": (size, size),
                "create_inputs": lambda size=size: self._create_graph_inputs(size, dtype)
            }
            for name, size in sizes
        ]
    
    def _create_graph_inputs(self, size: int, dtype: torch.dtype) -> Tuple[torch.Tensor, int]:
        adj_matrix = torch.zeros((size, size), device="cuda", dtype=dtype)
        
        mask = ~torch.eye(size, device="cuda", dtype=torch.bool)
        random_mask = torch.rand((size, size), device="cuda") < 0.5
        connection_mask = mask & random_mask
        
        weights = torch.randint(1, 11, (size, size), device="cuda", dtype=dtype)
        adj_matrix[connection_mask] = weights[connection_mask]
        
        if size > 1:
            forward_indices = torch.arange(size - 1, device="cuda")
            forward_weights = torch.randint(1, 6, (size - 1,), device="cuda", dtype=dtype)
            adj_matrix[forward_indices, forward_indices + 1] = forward_weights
        
        if size > 1:
            backward_mask = torch.rand(size - 1, device="cuda") < 0.3
            backward_indices = torch.arange(1, size, device="cuda")[backward_mask]
            backward_weights = torch.randint(1, 6, (backward_mask.sum(),), device="cuda", dtype=dtype)
            adj_matrix[backward_indices, backward_indices - 1] = backward_weights
        
        source = torch.randint(0, size, (1,), device="cuda").item()
        return adj_matrix, source

    def generate_sample(self, dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
        """
        Generate a single sample test case for debugging or interactive runs.
        
        Returns:
            A test case dictionary
        """
        name = "Sample (n = 4)"
        size = 4
        
        def create_sample_inputs():
            adj_matrix = torch.tensor([
                [0, 1, 4, 0],
                [0, 0, 2, 5],
                [0, 0, 0, 1],
                [0, 0, 0, 0]
            ], device="cuda", dtype=dtype)
            source = 0
            return adj_matrix, source
        
        return {
            "name": name,
            "dims": (size, size),
            "create_inputs": create_sample_inputs
        }
 
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the shortest path result is correct.
        
        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            
        Returns:
            Tuple of (is_correct, debug_info)
        """
        is_close = torch.allclose(actual_output, expected_output, rtol=1e-4, atol=1e-3)
        
        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            max_diff = torch.max(torch.abs(diff)).item()
            mean_diff = torch.mean(torch.abs(diff)).item()
            
            debug_info = {
                "max_difference": max_diff,
                "mean_difference": mean_diff,
                "expected": expected_output.cpu().numpy().tolist(),
                "actual": actual_output.cpu().numpy().tolist()
            }
        
        return is_close, debug_info
    
    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the shortest path solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # adjacency matrix
                ctypes.c_int,                    # source node
                ctypes.POINTER(ctypes.c_float),  # output distances
                ctypes.c_size_t,                 # N (number of nodes)
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
        return N * N * 2
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the number of nodes N and source node
        """
        N = test_case["dims"][0]
        # Source is passed as first parameter in the function signature
        return [N]
