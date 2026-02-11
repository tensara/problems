import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class matrix_multiplication(Problem):
    """Matrix multiplication problem."""
    
    is_exact = False
    
    def __init__(self):
        super().__init__(
            name="matrix-multiplication"
        )
    
    def reference_solution(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of matrix multiplication.
        
        Args:
            A: First input matrix
            B: Second input matrix
            
        Returns:
            Result of A * B
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=A.dtype):
            return torch.matmul(A, B)
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for matrix multiplication.
        
        Returns:
            List of test case dictionaries with varying matrix dimensions
        """
        # Matrix dimensions: (M, K) × (K, N) = (M, N)
        # dims represents (M, N, K)
        test_matrices = [
            {
                "name": "4096x4096 x 4096x4096",
                "dims": (4096, 4096, 4096),
            },
            {
                "name": "8192x8192 x 8192x4096",
                "dims": (8192, 4096, 8192),
            },
            {
                "name": "4096x4096 x 4096x8192",
                "dims": (4096, 8192, 4096),
            },
            {
                "name": "8192x8192 x 8192x8192",
                "dims": (8192, 8192, 8192),
            }
        ]
        
        test_cases = []
        for matrix in test_matrices:
            seed = Problem.get_seed(f"{self.name}_{matrix['name']}_{matrix['dims']}")
            test_cases.append({
                "name": matrix["name"],
                "dims": matrix["dims"],
                "create_inputs": lambda m=matrix["dims"], seed=seed, dtype=dtype: (
                    *(lambda g: (
                        torch.rand(m[0], m[2], device="cuda", dtype=dtype, generator=g) * 2 - 1,  # uniform [-1, 1]
                        torch.rand(m[2], m[1], device="cuda", dtype=dtype, generator=g) * 2 - 1   # uniform [-1, 1]
                    ))(torch.Generator(device="cuda").manual_seed(seed)),
                )
            })
        return test_cases
    
    def generate_sample(self, dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
        """
        Generate sample test case for matrix multiplication with predictable inputs.

        Returns:
            Dictionary containing the sample test case.
        """
        m_dims = (8, 8, 8)  # M, N, K dimensions
        return {
            "name": "8x8_square",
            "dims": m_dims,
            "create_inputs": lambda m_dims=m_dims: (
                torch.tensor([
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                    [17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
                    [25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0],
                    [33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0],
                    [41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0],
                    [49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0],
                    [57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0]
                ], device="cuda", dtype=dtype),
                torch.tensor([
                    [1.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0],
                    [-9.0, 10.0, -11.0, 12.0, -13.0, 14.0, -15.0, 16.0],
                    [17.0, -18.0, 19.0, -20.0, 21.0, -22.0, 23.0, -24.0],
                    [-25.0, 26.0, -27.0, 28.0, -29.0, 30.0, -31.0, 32.0],
                    [33.0, -34.0, 35.0, -36.0, 37.0, -38.0, 39.0, -40.0],
                    [-41.0, 42.0, -43.0, 44.0, -45.0, 46.0, -47.0, 48.0],
                    [49.0, -50.0, 51.0, -52.0, 53.0, -54.0, 55.0, -56.0],
                    [-57.0, 58.0, -59.0, 60.0, -61.0, 62.0, -63.0, 64.0]
                ], device="cuda", dtype=dtype)
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
            
            debug_info = {
                "max_difference": max_diff,
                "mean_difference": mean_diff
            }
        
        return is_close, debug_info
    
    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the matrix multiplication solution.
        
        IMPORTANT: Comments are required. Outline the FLOPs calculation.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # matrix_a
                ctypes.POINTER(ctypes.c_float),  # matrix_b
                ctypes.POINTER(ctypes.c_float),  # matrix_c (output)
                ctypes.c_size_t,                 # M (rows in A and C)
                ctypes.c_size_t,                 # N (columns in B and C)
                ctypes.c_size_t                  # K (columns in A, rows in B)
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
        # Matrix multiplication FLOPS = 2 * M * N * K
        # (One multiply and one add for each cell in the result, done K times)
        M, N, K = test_case["dims"]
        return 2 * M * N * K
    
    def get_mem(self, test_case: Dict[str, Any]) -> int:
        """
        Get the memory usage for the problem. Assumed to be all in DRAM
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            Memory usage in bytes
        """
        M, N, K = test_case["dims"]
        
        # Input: A (M*K) + B (K*N)
        # Output: C (M*N)
        dtype_bytes = 4  # 4 bytes per float32 element
        return (M * K + K * N + M * N) * dtype_bytes
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the dimensions M, N, K
        """
        M, N, K = test_case["dims"]
        return [M, N, K]