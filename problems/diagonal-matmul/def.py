import ctypes
import torch
from typing import List, Dict, Tuple, Any

from problem import Problem
from .solution import DiagonalMatmulSolutions

class diagonal_matmul(Problem, DiagonalMatmulSolutions):
    """Diagonal matrix multiplication problem."""
    
    def __init__(self):
        super().__init__(
            name="diagonal-matmul"
        )
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for diagonal matrix multiplication.
        
        Returns:
            List of test case dictionaries with varying matrix dimensions
        """
        # Diagonal matrix dimensions: (N,) and (N, M)
        test_matrices = [
            {
                "name": "2048 x 2048x2048",
                "dims": (2048, 2048),
            },
            {
                "name": "4096 x 4096x4096",
                "dims": (4096, 4096),
            },
            {
                "name": "6144 x 6144x6144",
                "dims": (6144, 6144),
            },
            {
                "name": "8192 x 8192x4096",
                "dims": (8192, 4096),
            }
        ]
        
        return [
            {
                "name": matrix["name"],
                "dims": matrix["dims"],
                "create_inputs": lambda m=matrix["dims"]: (
                    torch.rand(m[0], device="cuda", dtype=dtype) * 2 - 1,  # uniform [-1, 1]
                    torch.rand(m[0], m[1], device="cuda", dtype=dtype) * 2 - 1   # uniform [-1, 1]
                )
            }
            for matrix in test_matrices
        ]
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the diagonal matrix multiplication result is correct.
        
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
                "mean_difference": mean_diff
            }
        
        return is_close, debug_info
    
    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the diagonal matrix multiplication solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # diagonal_a
                ctypes.POINTER(ctypes.c_float),  # matrix_b
                ctypes.POINTER(ctypes.c_float),  # matrix_c (output)
                ctypes.c_size_t,                 # N (size of diagonal and rows in B)
                ctypes.c_size_t                  # M (columns in B and C)
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
        # Diagonal matrix multiplication FLOPS = N * M
        N, M = test_case["dims"]
        return N * M
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the dimensions N, M
        """
        N, M = test_case["dims"]
        return [N, M]
