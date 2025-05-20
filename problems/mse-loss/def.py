import torch
import ctypes
from typing import List, Dict, Tuple, Any
from problem import Problem
from .solution import MseLossSolutions

class mse_loss(Problem, MseLossSolutions):
    """Mean Squared Error loss problem."""
    
    def __init__(self):
        super().__init__(
            name="mse_loss"
        )
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for MSE loss function.
        
        Returns:
            List of test case dictionaries with varying sizes and dimensions
        """
        test_configs = [        
            (4096, 4096),             
            (8192, 8192),          
            (512, 512, 512),      
            (64, 64, 64, 64),       
            (32, 32, 32, 32, 32)    
        ]
        
        return [
            {
                "name": f"shape={shape}",
                "shape": shape,
                "create_inputs": lambda shape=shape: (
                    torch.randn(shape, device="cuda", dtype=dtype),     # Predictions
                    torch.randn(shape, device="cuda", dtype=dtype)      # Targets
                )
            }
            for shape in test_configs
        ]
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the MSE loss result is correct.
        
        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            
        Returns:
            Tuple of (is_correct, debug_info)
        """
        # Check if outputs are close (should be scalar values)
        is_close = torch.allclose(actual_output, expected_output, rtol=1e-5, atol=1e-5)
                
        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            abs_diff = torch.abs(diff).item()
            rel_diff = abs_diff / (torch.abs(expected_output).item() + 1e-8)
            
            debug_info = {
                "expected": expected_output.item(),
                "actual": actual_output.item(),
                "absolute_difference": abs_diff,
                "relative_difference": rel_diff
            }
        
        return is_close, debug_info
    
    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the MSE loss solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # predictions
                ctypes.POINTER(ctypes.c_float),  # targets
                ctypes.POINTER(ctypes.c_float),  # output (scalar)
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
            
        For MSE loss, we count:
        - One subtraction per element
        - One square per element
        - (n-1) additions for the sum (where n is the number of elements)
        - One division for the mean
        
        Returns:
            Number of floating point operations
        """
        shape = test_case["shape"]
        
        # Total elements in the tensor
        total_elements = 1
        for s in shape:
            total_elements *= s
            
        # FLOPs:
        # - total_elements subtractions
        # - total_elements multiplications (for squaring)
        # - (total_elements - 1) additions for sum
        # - 1 division for mean
        flops = (
            total_elements +          # subtraction
            total_elements +          # squaring
            (total_elements - 1) +    # sum
            1                         # divide (mean)
        )
        
        return flops
    
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