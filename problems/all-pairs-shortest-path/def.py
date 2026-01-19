import torch
import ctypes
from typing import List, Dict, Tuple, Any
import math

from problem import Problem

class all_pairs_shortest_path(Problem):
    """All-pairs shortest path problem using Floyd-Warshall algorithm."""
    
    is_exact = True
    
    def __init__(self):
        super().__init__(
            name="all-pairs-shortest-path"
        )
    
    def reference_solution(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of all-pairs shortest path using Floyd-Warshall.
        
        Args:
            adj_matrix: Weighted adjacency matrix (N x N) with positive weights
            
        Returns:
            Distance matrix with shortest paths between all pairs of nodes
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            N = adj_matrix.size(0)
            device = adj_matrix.device

            dist = adj_matrix.clone().float()
            mask = (adj_matrix == 0) & (torch.eye(N, device=device, dtype=torch.bool) == False)
            dist[mask] = float('inf')
            
            torch.diagonal(dist).fill_(0.0)
            
            for k in range(N):
                dist_through_k = dist[:, k:k+1] + dist[k:k+1, :]
                dist = torch.minimum(dist, dist_through_k)
            
            dist = torch.where(torch.isinf(dist), -1.0, dist)
            
            return dist

    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for all-pairs shortest path.
        
        Returns:
            List of test case dictionaries with varying graph sizes
        """
        sizes = [
            ("n = 512" , 512),
            ("n = 1024", 1024),
            ("n = 2048", 2048),
            ("n = 4096", 4096),
        ]
        
        test_cases = []
        for name, size in sizes:
            seed = Problem.get_seed(f"{self.name}_{name}_{(size, size)}")
            test_cases.append({
                "name": name,
                "dims": (size, size),
                "create_inputs": lambda size=size, seed=seed, dtype=dtype, create_inputs=self._create_graph_inputs: (
                    create_inputs(size, dtype, torch.Generator(device="cuda").manual_seed(seed))
                )
            })
        return test_cases
    
    def _create_graph_inputs(self, size: int, dtype: torch.dtype, generator: torch.Generator) -> Tuple[torch.Tensor]:
        """Create a connected directed graph with positive weights."""
        adj_matrix = torch.zeros((size, size), device="cuda", dtype=dtype)
        
        if size > 1:
            forward_indices = torch.arange(size - 1, device="cuda")
            forward_weights = torch.randint(1, 10, (size - 1,), device="cuda", dtype=dtype, generator=generator)
            adj_matrix[forward_indices, forward_indices + 1] = forward_weights
            
            # add some backward edges for cycles
            backward_mask = torch.rand(size - 1, device="cuda", generator=generator) < 0.4
            backward_indices = torch.arange(1, size, device="cuda")[backward_mask]
            backward_weights = torch.randint(1, 15, (backward_mask.sum(),), device="cuda", dtype=dtype, generator=generator)
            adj_matrix[backward_indices, backward_indices - 1] = backward_weights
        
        mask = ~torch.eye(size, device="cuda", dtype=torch.bool)
        random_mask = torch.rand((size, size), device="cuda", generator=generator) < 0.3  # 30% connectivity
        connection_mask = mask & random_mask
        
        weights = torch.randint(1, 20, (size, size), device="cuda", dtype=dtype, generator=generator)
        adj_matrix[connection_mask] = weights[connection_mask]
        
        # making disconnected graph for the smaller testcase        
        if size == 512:
            num_disconnected = torch.randint(size // 10, size // 5, (1,), device="cuda", generator=generator).item()
            disconnected_nodes = torch.randperm(size, device="cuda", generator=generator)[:num_disconnected]
            adj_matrix[disconnected_nodes, :] = 0
            adj_matrix[:, disconnected_nodes] = 0
        
        return (adj_matrix,)

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
                [0, 3, 8, 0],
                [0, 0, 0, 1], 
                [0, 4, 0, 0],
                [2, 0, 0, 0]
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
        Verify if the all-pairs shortest path result is correct.
        
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
                "expected_shape": expected_output.shape,
                "actual_shape": actual_output.shape
            }
        
        return is_close, debug_info
    
    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the all-pairs shortest path solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # input adjacency matrix
                ctypes.POINTER(ctypes.c_float),  # output distance matrix
                ctypes.c_size_t,                 # N (number of nodes)
            ],
            "restype": None
        }
    

    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the number of nodes N
        """
        N = test_case["dims"][0]
        return [N] 