import torch
import ctypes
from typing import List, Dict, Tuple, Any
import math

from problem import Problem

class min_spanning_tree(Problem):
    """Minimum spanning tree problem using parallel Prim's algorithm."""
    
    is_exact = True
    
    def __init__(self):
        super().__init__(
            name="min-spanning-tree"
        )
    
    def reference_solution(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of minimum spanning tree using parallel Prim's algorithm.
        
        Args:
            adj_matrix: Weighted adjacency matrix (N x N) with positive integer weights
            
        Returns:
            Total weight of minimum spanning tree as a scalar tensor
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=adj_matrix.dtype):
            N = adj_matrix.size(0)
            device = adj_matrix.device
            
            if N <= 1:
                return torch.tensor(0.0, device=device)
            
            adj_float = adj_matrix.float()
            adj_float = torch.where(adj_float == 0, float('inf'), adj_float)
            
            in_mst = torch.zeros(N, device=device, dtype=torch.bool)
            min_edge_weight = torch.full((N,), float('inf'), device=device)
            
            in_mst[0] = True
            min_edge_weight = adj_float[0].clone() 
            min_edge_weight[0] = 0.0
            
            mst_weight = 0.0
            
            for _ in range(N - 1):
                masked_weights = torch.where(in_mst, float('inf'), min_edge_weight)
                
                min_idx = torch.argmin(masked_weights)
                min_weight = masked_weights[min_idx]
                
                if min_weight == float('inf'):
                    return torch.tensor(float('inf'), device=device)
                
                in_mst[min_idx] = True
                mst_weight += min_weight.item()
                
                new_distances = adj_float[min_idx]
                min_edge_weight = torch.minimum(min_edge_weight, new_distances)
            
            return torch.tensor(mst_weight, device=device)

    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for minimum spanning tree.
        
        Returns:
            List of test case dictionaries with varying graph sizes
        """
        sizes = [
            ("n = 1024", 1024),
            ("n = 2048", 2048),
            ("n = 4096", 4096),
            ("n = 6144", 6144)
        ]
        
        test_cases = []
        for name, size in sizes:
            seed = Problem.get_seed(f"{self.name}_{name}")
            
            def create_inputs_closure(size=size, seed=seed, dtype=dtype, self_ref=self):
                return (self_ref._create_graph_inputs(size, dtype, seed),)
            
            test_cases.append({
                "name": name,
                "dims": (size, size),
                "create_inputs": create_inputs_closure
            })
        
        return test_cases
    
    def _create_graph_inputs(self, size: int, dtype: torch.dtype, seed: int = None) -> torch.Tensor:
        """Create a connected undirected graph."""
        gen = torch.Generator(device="cuda").manual_seed(seed) if seed is not None else None
        adj_matrix = torch.zeros((size, size), device="cuda", dtype=dtype)
        
        for i in range(1, size):
            parent = torch.randint(0, i, (1,), device="cuda", generator=gen).item()
            weight = torch.randint(1, 11, (1,), device="cuda", dtype=dtype, generator=gen).item()
            adj_matrix[parent, i] = weight
            adj_matrix[i, parent] = weight
        
        upper_triu_mask = torch.triu(torch.ones(size, size, device="cuda", dtype=torch.bool), diagonal=1)
        existing_edges = adj_matrix > 0
        available_positions = upper_triu_mask & ~existing_edges
        
        if available_positions.any():
            num_additional = min(size * 2, available_positions.sum().item())
            available_indices = torch.where(available_positions)
            perm = torch.randperm(len(available_indices[0]), device="cuda", generator=gen)[:num_additional]
            
            u_add = available_indices[0][perm]
            v_add = available_indices[1][perm]
            weights_add = torch.randint(1, 21, (num_additional,), device="cuda", dtype=dtype, generator=gen)
            
            adj_matrix[u_add, v_add] = weights_add
            adj_matrix[v_add, u_add] = weights_add
        
        return adj_matrix

    def generate_sample(self, dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
        """
        Generate a single sample test case for debugging or interactive runs.
        
        Returns:
            A test case dictionary
        """
        name = "Sample (n = 4)"
        size = 4
        
        adj_matrix = torch.tensor([
            [0, 2, 0, 6],
            [2, 0, 3, 8],
            [0, 3, 0, 5],
            [6, 8, 5, 0]
        ], device="cuda", dtype=dtype)
        
        def create_sample_inputs():
            return (adj_matrix,)
        
        return {
            "name": name,
            "dims": (size, size),
            "create_inputs": create_sample_inputs
        }

    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the minimum spanning tree result is correct.
        
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
            abs_diff = torch.abs(diff).item()
            
            debug_info = {
                "difference": abs_diff,
                "expected": expected_output.item(),
                "actual": actual_output.item()
            }
        
        return is_close, debug_info
    
    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the minimum spanning tree solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # adjacency matrix
                ctypes.POINTER(ctypes.c_float),  # output MST weight
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