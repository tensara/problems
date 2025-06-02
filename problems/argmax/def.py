import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem

class argmax(Problem):
    """Argmax over dimension problem."""
    
    def __init__(self):
        super().__init__(
            name="argmax"
        )
    
    def reference_solution(self, input_tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """
        PyTorch implementation of argmax over dimension.
        
        Args:
            input_tensor: Input tensor of arbitrary shape
            dim: Dimension to perform argmax over
            
        Returns:
            Result of argmax operation
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            return torch.argmax(input_tensor, dim=dim).to(torch.int32)
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for argmax over dimension.
        
        Returns:
            List of test case dictionaries with varying sizes and dimensions
        """
        test_configs = [
            # (shape, dim)
            ((16, 128, 256), 1),
            ((32, 512, 512), 0),
            ((8, 1024, 1024), 2),
            ((64, 128, 128, 128), 2),
            ((4, 256, 256, 256), 1),
            ((128, 64, 64, 64), 3)
        ]
        
        return [
            {
                "name": f"shape={shape}, dim={dim}",
                "shape": shape,
                "dim": dim,
                "create_inputs": lambda shape=shape, dim=dim: (
                    torch.rand(shape, device="cuda", dtype=dtype) * 10.0 - 5.0,  # uniform [-5, 5]
                    dim
                )
            }
            for shape, dim in test_configs
        ]
    
    def generate_sample(self, dtype: torch.dtype = torch.float32) -> List[Dict[str, Any]]:
        """
        Generate a single sample test case for debugging or interactive runs.
        
        Returns:
            A list containing a single test case dictionary
        """
        shape = (8,8)
        dim = 0
        return [
            {
                "name": f"shape={shape}, dim={dim}",
                "shape": shape,
                "dim": dim,
                "create_inputs": lambda shape=shape, dim=dim: (
                    torch.tensor([[4.0, -4.5, 5.0, 3.0, -2.0, 3.2, 2.2, 0.5],
                                [-2.0, -4.5, 5.0, 3.0, -2.0, 3.2, 2.2, 0.5],
                                [4.0, -4.5, 5.0, 3.0, -2.0, 3.2, 2.2, 0.5],
                                [4.0, -4.5, 5.0, 3.0, -2.0, 3.2, 2.2, 0.5],
                                [4.0, -4.5, 5.0, 3.0, -2.0, 3.2, 2.2, 0.5],
                                [4.0, -4.5, 5.0, 3.0, -2.0, 3.2, 2.2, 0.5],
                                [4.0, -4.5, 5.0, 3.0, -2.0, 3.2, 2.2, 0.5],
                                [4.0, -4.5, 5.0, 3.0, -2.0, 3.2, 2.2, 0.5]], device="cuda", dtype=dtype),
                    dim
                )
            }
        ]

    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the argmax result is correct.
        
        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            
        Returns:
            Tuple of (is_correct, debug_info)
        """
        is_equal = torch.all(actual_output == expected_output)
        
        debug_info = {}
        if not is_equal:
            diff_mask = actual_output != expected_output
            
            # Find indices of differences
            diff_indices = torch.nonzero(diff_mask)
            sample_diffs = {}
            
            # Get up to 5 sample differences
            for i in range(min(5, len(diff_indices))):
                idx = tuple(diff_indices[i].tolist())
                sample_diffs[f"{i}"] = {
                    "expected": expected_output[idx].item(),
                    "actual": actual_output[idx].item(),
                    "diff": abs(actual_output[idx].item() - expected_output[idx].item())
                }
            
            debug_info = {
                "max_difference": torch.max(torch.abs(actual_output - expected_output)).item(),
                "sample_differences": sample_diffs
            }
        
        return is_equal, debug_info
    
    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the argmax over dimension solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # input_tensor
                ctypes.c_int32,                    # dim
                ctypes.POINTER(ctypes.c_int32),  # output
                ctypes.POINTER(ctypes.c_int32), # shape
                ctypes.c_int32,                 # ndim
            ],
            "restype": None
        }
    
    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the number of floating point operations for the problem.
        
        Args:
            test_case: The test case dictionary
            
        For argmax, we count:
        - One comparison per element being compared
        
        Returns:
            Number of floating point operations
        """
        shape = test_case["shape"]
        dim = test_case["dim"]
        
        # Total elements in the tensor
        total_elements = 1
        for s in shape:
            total_elements *= s
            
        # Number of elements being compared (size of reduction dimension)
        reduce_size = shape[dim]
        
        # Number of reduction operations
        num_reductions = total_elements // reduce_size
        
        # Each reduction requires (reduce_size - 1) comparisons
        return num_reductions * (reduce_size - 1)
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the shape array and number of dimensions
        """
        return [
            torch.tensor(list(test_case["shape"]), dtype=torch.int32),
            len(test_case["shape"]),
        ]
