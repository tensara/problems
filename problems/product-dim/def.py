import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem
from .solution import ProductDimSolutions


class product_dim(ProductDimSolutions, Problem):
    """Product over dimension problem."""
    
    def __init__(self):
        super().__init__(
            name="product-dim"
        )
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for product over dimension.
        
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
                    torch.rand(shape, device="cuda", dtype=dtype) * 1.0 - 0.5,  # uniform [-0.5, 0.5]
                    dim
                )
            }
            for shape, dim in test_configs
        ]
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the product reduction result is correct.
        
        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            
        Returns:
            Tuple of (is_correct, debug_info)
        """
        is_close = torch.allclose(actual_output, expected_output, rtol=1e-3, atol=1e-3)
        
        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            max_diff = torch.max(torch.abs(diff)).item()
            mean_diff = torch.mean(torch.abs(diff)).item()
            
            # Find indices of largest differences
            flat_diff = diff.flatten()
            _, top_indices = torch.topk(torch.abs(flat_diff), min(5, flat_diff.numel()))
            
            # Get sample differences
            sample_diffs = {}
            for i, idx in enumerate(top_indices):
                sample_diffs[f"{i}"] = {
                    "expected": expected_output.flatten()[idx].item(),
                    "actual": actual_output.flatten()[idx].item(),
                    "diff": diff.flatten()[idx].item()
                }
            
            debug_info = {
                "max_difference": max_diff,
                "mean_difference": mean_diff,
                "sample_differences": sample_diffs
            }
        
        return is_close, debug_info
    
    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the product over dimension solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # input_tensor
                ctypes.c_int,                    # dim
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.POINTER(ctypes.c_size_t), # shape
                ctypes.c_size_t,                 # ndim
            ],
            "restype": None
        }
    
    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the number of floating point operations for the problem.
        
        Args:
            test_case: The test case dictionary
            
        For product reduction, we count:
        - One multiplication per element being reduced
        
        Returns:
            Number of floating point operations
        """
        shape = test_case["shape"]
        dim = test_case["dim"]
        
        # Total elements in the tensor
        total_elements = 1
        for s in shape:
            total_elements *= s
            
        # Number of elements being reduced (size of reduction dimension)
        reduce_size = shape[dim]
        
        # Number of reduction operations
        num_reductions = total_elements // reduce_size
        
        # Each reduction requires (reduce_size - 1) multiplications
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
            torch.tensor(list(test_case["shape"]), dtype=torch.int64),
            len(test_case["shape"]),
        ]
