import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class elu(Problem):
    """ELU (Exponential Linear Unit) activation function problem."""
    
    def __init__(self):
        super().__init__(
            name="elu"
        )
        self.alpha = 1.0  # Default alpha value for ELU
    
    def reference_solution(self, input_matrix: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of ELU.
        
        Args:
            input_matrix: Input matrix of shape (M, N)
            
        Returns:
            Result of ELU activation
        """
        with torch.no_grad():
            return torch.nn.functional.elu(input_matrix, alpha=self.alpha)
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for ELU.
        
        Returns:
            List of test case dictionaries with varying sizes
        """
        # Test case configurations with specific matrix sizes
        test_configs = [
            ("4096x4096", 4096, 4096),
            ("6144x4096", 6144, 4096),
            ("4096x7168", 4096, 7168),
            ("4096x8192", 4096, 8192),
            ("8192x8192", 8192, 8192)
        ]
        
        test_cases = []
        for name, m, n in test_configs:
            seed = Problem.get_seed(f"{self.name}_{name}_{(m, n)}")
            test_cases.append({
                "name": name,
                "rows": m,
                "cols": n,
                "create_inputs": lambda m=m, n=n, seed=seed, dtype=dtype: (
                    *(lambda g: (
                        torch.rand((m, n), device="cuda", dtype=dtype, generator=g) * 10.0 - 5.0,  # uniform [-5, 5]
                    ))(torch.Generator(device="cuda").manual_seed(seed)),
                )
            })
        return test_cases
    
    def generate_sample(self, dtype: torch.dtype = torch.float32) -> List[Dict[str, Any]]:
        """
        Generate a single sample test case for debugging or interactive runs.
        
        Returns:
            A list containing a single test case dictionary
        """
        m, n = (4, 4)
        return {
            "name": f"Sample ({m}x{n})",
            "rows": m,
            "cols": n,
            "create_inputs": lambda m=m, n=n: (
                torch.tensor([
                    [-5.0, -2.5, 0.0, 2.5],
                    [-4.0, -1.5, 1.0, 3.5], 
                    [-3.0, -0.5, 2.0, 4.5],
                    [-2.0, 0.5, 3.0, 5.0]
                ], device="cuda", dtype=dtype),
            )
        }

    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the ELU result is correct.
        
        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            
        Returns:
            Tuple of (is_correct, debug_info)
        """
        is_close = torch.allclose(actual_output, expected_output, rtol=1e-5, atol=1e-5)
        
        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            max_diff = torch.max(torch.abs(diff)).item()
            mean_diff = torch.mean(torch.abs(diff)).item()
            
            # Find indices of largest differences
            flat_diff = diff.flatten()
            _, top_indices = torch.topk(torch.abs(flat_diff), min(5, flat_diff.numel()))
            
            # Convert flat indices back to 2D coordinates
            m, n = expected_output.shape
            sample_diffs = {}
            for i, idx in enumerate(top_indices):
                row = idx.item() // n
                col = idx.item() % n
                sample_diffs[f"({row}, {col})"] = {
                    "expected": expected_output[row, col].item(),
                    "actual": actual_output[row, col].item(),
                    "diff": diff[row, col].item()
                }
            
            debug_info = {
                "max_difference": max_diff,
                "mean_difference": mean_diff,
                "sample_differences": sample_diffs
            }
        
        return is_close, debug_info
    
    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the ELU solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # input_matrix
                ctypes.POINTER(ctypes.c_float),  # output_matrix
                ctypes.c_size_t,                 # rows (M)
                ctypes.c_size_t,                 # columns (N)
                ctypes.c_float                   # alpha
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
        # Extract dimensions from test case
        M = test_case["rows"]
        N = test_case["cols"]
        
        # M*N FLOPs:
        # - Each element requires 1 comparison operation
        # - For negative values: 1 multiply (alpha), 1 exponentiation, 1 subtract (exp-1)
        # - We approximate this as 3 FLOPs per element (worst case)
        return 3 * M * N
    
    def get_mem(self, test_case: Dict[str, Any]) -> int:
        """
        Get the memory usage for the problem. Assumed to be all in DRAM
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            Memory usage in bytes
        """
        M = test_case["rows"]
        N = test_case["cols"]
        
        # Input: M*N elements, Output: M*N elements (same size)
        dtype_bytes = 4  # 4 bytes per float32 element
        return (M * N + M * N) * dtype_bytes
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the rows M, columns N, and alpha value
        """
        M = test_case["rows"]
        N = test_case["cols"]
        return [M, N, self.alpha]
