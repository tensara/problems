import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem

class cumsum(Problem):
    """Cumulative sum (prefix sum) problem."""
    
    def __init__(self):
        super().__init__(
            name="cumsum"
        )
    
    def reference_solution(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of cumulative sum.
        
        Args:
            input_tensor: Input tensor of shape (N)
            
        Returns:
            Cumulative sum of the input tensor
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            return torch.cumsum(input_tensor, dim=0)
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for cumulative sum.
        
        Returns:
            List of test case dictionaries with varying sizes
        """
        
        test_configs = [
            65536,
            131072,
            262144,
            524288,
            1048576,
        ]
        
        return [
            {
                "name": f"N={size}",
                "size": size,
                "create_inputs": lambda s=size: (
                    torch.rand(s, device="cuda", dtype=dtype) * 10.0 - 5.0,
                )
            }
            for size in test_configs
        ]
    
    def generate_sample(self, dtype: torch.dtype = torch.float32) -> List[Dict[str, Any]]:
        """
        Generate a single sample test case for debugging or interactive runs.
        
        Returns:
            A list containing a single test case dictionary
        """
        size = 8 # Sample size
        return [
            {
                "name": f"N={size}",
                "size": size,
                "create_inputs": lambda s=size: (
                    torch.arange(1, s + 1, device="cuda", dtype=dtype).float(), # Sequential input for easy verification
                )
            }
        ]
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the cumulative sum result is correct.
        
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
            
            # Find indices of largest differences
            _, top_indices = torch.topk(torch.abs(diff), min(5, diff.numel()))
            
            sample_diffs = {
                f"{idx.item()}": {
                    "expected": expected_output[idx].item(),
                    "actual": actual_output[idx].item(),
                    "diff": diff[idx].item()
                }
                for idx in top_indices
            }
            
            debug_info = {
                "max_difference": max_diff,
                "mean_difference": mean_diff,
                "sample_differences": sample_diffs
            }
        
        return is_close, debug_info
    
    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the cumulative sum solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # input_tensor
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_size_t,                 # size (N)
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
        # For each element (except the first), we need one addition
        N = test_case["size"]
        return N - 1
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the input size N
        """
        N = test_case["size"]
        return [N]
