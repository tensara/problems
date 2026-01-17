import torch
import ctypes
from typing import List, Dict, Tuple, Any
from problem import Problem

class mse_loss(Problem):
    """Mean Squared Error loss problem."""
    
    def __init__(self):
        super().__init__(
            name="mse_loss"
        )
    
    def reference_solution(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of MSE loss function.
        
        Args:
            predictions: Predicted values tensor of arbitrary shape
            targets: Target values tensor of the same shape as predictions
            
        Returns:
            Mean squared error loss as a scalar tensor
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            return torch.mean((predictions - targets) ** 2)
    
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
        
        test_cases = []
        for shape in test_configs:
            seed = Problem.get_seed(f"{self.name}_shape={shape}")
            test_cases.append({
                "name": f"shape={shape}",
                "shape": shape,
                "create_inputs": lambda shape=shape, seed=seed, dtype=dtype: (
                    *(lambda g: (
                        torch.randn(shape, device="cuda", dtype=dtype, generator=g),     # Predictions
                        torch.randn(shape, device="cuda", dtype=dtype, generator=g)      # Targets
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
        shape = (8, 8)  # Sample dimensions
        return {
            "name": f"shape={shape}",
            "shape": shape,
            "create_inputs": lambda shape=shape: (
                torch.randn(shape, device="cuda", dtype=dtype),     # Predictions
                torch.randn(shape, device="cuda", dtype=dtype)      # Targets
            )
        }
    
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
    
    def get_mem(self, test_case: Dict[str, Any]) -> int:
        """
        Get the memory usage for the problem. Assumed to be all in DRAM
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            Memory usage in bytes
        """
        shape = test_case["shape"]
        
        # Total elements in the tensor
        total_elements = 1
        for s in shape:
            total_elements *= s
        
        # Naive MSE loss:
        # 1. Read predictions → total_elements
        # 2. Read targets → total_elements
        # 3. Write diff = predictions - targets → total_elements (materialized)
        # 4. Read diff → total_elements
        # 5. Write squared = diff^2 → total_elements (materialized)
        # 6. Read squared → total_elements
        # 7. Write sum → 1 (materialized)
        # 8. Read sum → 1
        # 9. Write mean → 1
        
        dtype_bytes = 4  # 4 bytes per float32 element
        return (2 * total_elements +      # read predictions, targets
                2 * total_elements +      # write and read diff
                2 * total_elements +      # write and read squared
                2) * dtype_bytes           # write and read sum, write mean
    
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