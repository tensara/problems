import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class matmul_3d(Problem):
    """3D matrix multiplication problem."""
    
    def __init__(self):
        super().__init__(
            name="matmul-3d"
        )
    
    def reference_solution(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of 3D tensor-matrix multiplication.
        
        Args:
            A: First input tensor of shape (N, M, K)
            B: Second input matrix of shape (K, L)
            
        Returns:
            Result of shape (N, M, L) from multiplying A and B along the last dimension of A
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            return torch.matmul(A, B)
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for 3D tensor-matrix multiplication.
        
        Returns:
            List of test case dictionaries with varying dimensions
        """
        # Matrix dimensions: (N, M, K) × (K, L) = (N, M, L)
        # dims represents (N, M, K, L)
        test_matrices = [
            {
                "name": "32x4096x4096 x 4096x4096",
                "dims": (32, 4096, 4096, 4096),
            },
            {
                "name": "16x8192x8192 x 8192x4096",
                "dims": (16, 8192, 8192, 4096),
            },
            {
                "name": "64x4096x4096 x 4096x8192",
                "dims": (64, 4096, 4096, 8192),
            },
            {
                "name": "8x8192x8192 x 8192x8192",
                "dims": (8, 8192, 8192, 8192),
            }
        ]
        
        return [
            {
                "name": matrix["name"],
                "dims": matrix["dims"],
                "create_inputs": lambda m=matrix["dims"]: (
                    torch.rand(m[0], m[1], m[2], device="cuda", dtype=dtype) * 2 - 1,  # A: (N,M,K)
                    torch.rand(m[2], m[3], device="cuda", dtype=dtype) * 2 - 1         # B: (K,L)
                )
            }
            for matrix in test_matrices
        ]
    
    def generate_sample(self, dtype: torch.dtype = torch.float32) -> List[Dict[str, Any]]:
        """
        Generate a single sample test case for debugging or interactive runs.
        
        Returns:
            A list containing a single test case dictionary
        """
        N, M, K, L = (4, 4, 4, 4)
        return {
            "name": f"Sample ({N}x{M}x{K} * {K}x{L})",
            "dims": (N, M, K, L),
            "create_inputs": lambda n=N, m=M, k=K, l=L: (
                torch.arange(1, n*m*k + 1, device="cuda", dtype=dtype).float().view(n, m, k),
                torch.arange(1, k*l + 1, device="cuda", dtype=dtype).float().view(k, l)
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
        Get the function signature for 3D tensor-matrix multiplication.
        
        IMPORTANT: Comments are required. Outline the FLOPs calculation.
        
        For 3D tensor-matrix multiplication:
        - Input A has shape (N, M, K)
        - Input B has shape (K, L)
        - Output C has shape (N, M, L)
        - For each of the N*M output elements, we perform K multiply-adds
        - Total FLOPs = 2 * N * M * K * L
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # matrix_a (N, M, K)
                ctypes.POINTER(ctypes.c_float),  # matrix_b (K, L)
                ctypes.POINTER(ctypes.c_float),  # matrix_c (output) (N, M, L)
                ctypes.c_size_t,                 # N (first dim of A)
                ctypes.c_size_t,                 # M (second dim of A)
                ctypes.c_size_t,                 # K (third dim of A, first dim of B)
                ctypes.c_size_t                  # L (second dim of B)
            ],
            "restype": None
        }
    
    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the number of floating point operations for 3D tensor-matrix multiplication.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            Number of floating point operations
        """
        # 3D tensor-matrix multiplication FLOPS = 2 * N * M * K * L
        # (One multiply and one add for each cell in the result, done K times, for N*M elements)
        N, M, K, L = test_case["dims"]
        return 2 * N * M * K * L
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the dimensions N, M, K, L
        """
        N, M, K, L = test_case["dims"]
        return [N, M, K, L]