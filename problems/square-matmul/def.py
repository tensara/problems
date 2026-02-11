import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class square_matmul(Problem):
    """Square matrix multiplication problem."""
    
    is_exact = False
    
    def __init__(self):
        super().__init__(
            name="square-matmul"
        )
    
    def reference_solution(self, matrix_a: torch.Tensor, matrix_b: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of square matrix multiplication.
        
        Args:
            matrix_a: First input matrix of shape (N, N)
            matrix_b: Second input matrix of shape (N, N)
            
        Returns:
            Result of matrix multiplication of shape (N, N)
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=matrix_a.dtype):
            return torch.matmul(matrix_a, matrix_b)
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for square matrix multiplication.
        
        Returns:
            List of test case dictionaries with varying sizes
        """
        # Test case configurations with specific matrix sizes
        matrix_sizes = [4096, 6144, 7168, 8192, 9216]
        
        test_cases = []
        for n in matrix_sizes:
            seed = Problem.get_seed(f"{self.name}_N={n}")
            test_cases.append({
                "name": f"{n}x{n}",
                "size": n,
                "create_inputs": lambda n=n, seed=seed, dtype=dtype: (
                    *(lambda g: (
                        torch.rand((n, n), device="cuda", dtype=dtype, generator=g) * 2 - 1,  # uniform [-1, 1]
                        torch.rand((n, n), device="cuda", dtype=dtype, generator=g) * 2 - 1   # uniform [-1, 1]
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
        n = 4 
        return {
            "name": f"{n}x{n}",
            "size": n,
            "create_inputs": lambda n=n: (
                torch.randn((n, n), device="cuda", dtype=dtype) * 2 - 1,
                torch.randn((n, n), device="cuda", dtype=dtype) * 2 - 1
            )
        }
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the matrix multiplication result is correct.
        
        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            
        Returns:
            Tuple of (is_correct, debug_info)
        """
        is_close = torch.allclose(actual_output, expected_output, rtol=2e-4, atol=5e-3)
        
        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            max_diff = torch.max(torch.abs(diff)).item()
            mean_diff = torch.mean(torch.abs(diff)).item()
            
            # Find indices of largest differences
            flat_diff = diff.flatten()
            _, top_indices = torch.topk(torch.abs(flat_diff), min(5, flat_diff.numel()))
            
            # Convert flat indices back to 2D coordinates
            n = expected_output.shape[0]
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
        Get the function signature for the square matrix multiplication solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # matrix_a
                ctypes.POINTER(ctypes.c_float),  # matrix_b
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_size_t                  # size (N)
            ],
            "restype": None
        }
    
    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the number of floating point operations for the problem.
        
        Args:
            test_case: The test case dictionary
            
        IMPORTANT: Comments are required. Outline the FLOPs calculation.
        
        Returns:
            Number of floating point operations
        """
        # Extract dimension from test case
        N = test_case["size"]
        
        # N*N*N*2 FLOPs:
        # - Each output element requires N MAD operations
        # - Each MAD (Multiply-Add) counts as 2 FLOPs
        # - There are N*N output elements
        return N * N * N * 2
    
    def get_mem(self, test_case: Dict[str, Any]) -> int:
        """
        Get the memory usage for the problem. Assumed to be all in DRAM
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            Memory usage in bytes
        """
        N = test_case["size"]
        
        # Input: A (N*N) + B (N*N)
        # Output: C (N*N)
        dtype_bytes = 4  # 4 bytes per float32 element
        return (N * N + N * N + N * N) * dtype_bytes
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the matrix size N
        """
        N = test_case["size"]
        return [N]