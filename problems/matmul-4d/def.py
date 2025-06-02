import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class matmul_4d(Problem):
    """4D tensor-matrix multiplication problem."""
    
    def __init__(self):
        super().__init__(
            name="matmul-4d"
        )
    
    def reference_solution(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of 4D tensor-matrix multiplication.
        
        Args:
            A: First input tensor of shape (b, i, j, l)
            B: Second input matrix of shape (l, k)
            
        Returns:
            Result of shape (b, i, j, k) from multiplying A and B along the last dimension of A
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            return torch.einsum("bijl,lk->bijk", A, B)
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for 4D tensor-matrix multiplication.
        
        Returns:
            List of test case dictionaries with varying dimensions
        """
        # Tensor dimensions: (b, i, j, l) × (l, k) = (b, i, j, k)
        # dims represents (b, i, j, l, k)
        test_matrices = [
            {
                "name": "16x256x512x256 x 256x768",
                "dims": (16, 256, 512, 256, 768),
            },
            {
                "name": "8x128x256x128 x 128x512",
                "dims": (8, 128, 256, 128, 512),
            },
            {
                "name": "32x64x128x64 x 64x256",
                "dims": (32, 64, 128, 64, 256),
            },
            {
                "name": "4x32x64x32 x 32x128",
                "dims": (4, 32, 64, 32, 128),
            }
        ]
        
        return [
            {
                "name": matrix["name"],
                "dims": matrix["dims"],
                "create_inputs": lambda m=matrix["dims"]: (
                    torch.rand(m[0], m[1], m[2], m[3], device="cuda", dtype=dtype) * 2 - 1,  # A: (b,i,j,l)
                    torch.rand(m[3], m[4], device="cuda", dtype=dtype) * 2 - 1         # B: (l,k)
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
        b, i, j, l, k = (2, 2, 2, 3, 4) # Sample dimensions
        return [
            {
                "name": f"Sample ({b}x{i}x{j}x{l} * {l}x{k})",
                "dims": (b, i, j, l, k),
                "create_inputs": lambda b=b, i=i, j=j, l=l, k_dim=k: (
                    torch.arange(1, b*i*j*l + 1, device="cuda", dtype=dtype).float().view(b, i, j, l),
                    torch.arange(1, l*k_dim + 1, device="cuda", dtype=dtype).float().view(l, k_dim)
                )
            }
        ]
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the tensor-matrix multiplication result is correct.
        
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
        Get the function signature for 4D tensor-matrix multiplication.
        
        IMPORTANT: Comments are required. Outline the FLOPs calculation.
        
        For 4D tensor-matrix multiplication:
        - Input A has shape (b, i, j, l)
        - Input B has shape (l, k)
        - Output C has shape (b, i, j, k)
        - For each of the b*i*j output elements, we perform l multiply-adds
        - Total FLOPs = 2 * b * i * j * l * k
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # tensor_a (b, i, j, l)
                ctypes.POINTER(ctypes.c_float),  # matrix_b (l, k)
                ctypes.POINTER(ctypes.c_float),  # tensor_c (output) (b, i, j, k)
                ctypes.c_size_t,                 # b (first dim of A)
                ctypes.c_size_t,                 # i (second dim of A)
                ctypes.c_size_t,                 # j (third dim of A)
                ctypes.c_size_t,                 # l (fourth dim of A, first dim of B)
                ctypes.c_size_t                  # k (second dim of B)
            ],
            "restype": None
        }
    
    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the number of floating point operations for 4D tensor-matrix multiplication.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            Number of floating point operations
        """
        # 4D tensor-matrix multiplication FLOPS = 2 * b * i * j * l * k
        # (One multiply and one add for each cell in the result, done l times, for b*i*j elements)
        b, i, j, l, k = test_case["dims"]
        return 2 * b * i * j * l * k
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the dimensions b, i, j, l, k
        """
        b, i, j, l, k = test_case["dims"]
        return [b, i, j, l, k]
