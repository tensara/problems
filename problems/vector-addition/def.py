import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem

class vector_addition(Problem):
    """Vector addition problem."""
    
    def __init__(self):
        super().__init__(
            name="vector-addition"
        )
    
    def reference_solution(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of vector addition.
        
        Args:
            A: First input tensor
            B: Second input tensor
            
        Returns:
            Result of A + B
        """
        with torch.no_grad():
            return A + B
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for vector addition.
        
        Returns:
            List of test case dictionaries with varying sizes
        """
        sizes = [
            ("n = 2^20", 1048576),
            ("n = 2^22", 4194304),
            ("n = 2^23", 8388608),
            ("n = 2^25", 33554432),
            ("n = 2^26", 67108864),
            ("n = 2^29", 536870912),
            ("n = 2^30", 1073741824),
        ]
        
        return [
            {
                "name": name,
                "dims": (size,),
                "create_inputs": lambda size=size: (
                    torch.rand(size, device="cuda", dtype=dtype) * 2 - 1,  # uniform [-1, 1]
                    torch.rand(size, device="cuda", dtype=dtype) * 2 - 1   # uniform [-1, 1]
                )
            }
            for name, size in sizes
        ]
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the vector addition result is correct.
        
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
            
            debug_info = {
                "max_difference": max_diff,
                "mean_difference": mean_diff
            }
        
        return is_close, debug_info
    
    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the vector addition solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # input_a
                ctypes.POINTER(ctypes.c_float),  # input_b
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_size_t                  # N
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
        # Vector addition has 1 FLOP per element
        N = test_case["dims"][0]
        return N
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the vector length N
        """
        N = test_case["dims"][0]
        return [N]