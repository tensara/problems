import torch
import torch.nn as nn
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem 

class rms_norm(Problem):
    """RMS Normalization problem."""

    def __init__(self):
        super().__init__(
            name="rms-norm"
        )
        self.epsilon = 1e-5  # Standard epsilon for RMSNorm

    def reference_solution(self, x: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of RMS Normalization.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            # Calculate the RMS along the feature dimension
            # For 2D inputs (batch_size, num_features), this is along dim=1
            # For higher dimensional inputs, still use dim=1 (the feature dimension)
            rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)
            
            # Normalize the input by dividing by the RMS
            output = x / rms
            
            return output

    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for RMS Normalization.

        Returns:
            List of test case dictionaries with varying sizes
        """
        
        # Define configurations: (batch_size, features, *dims)
        test_configs = [
            (1024, 1024),                   # 2D tensor (batch, features)
            (1024, 4096),                  # 2D tensor, larger size
            (2048, 8192),                  # 2D tensor, typical for transformer models
            (512, 16384)            
        ]

        return [
            {
                "name": f"shape={shape}",
                "shape": shape,
                "create_inputs": lambda shape=shape: (
                    torch.randn(*shape, device="cuda", dtype=dtype),  # Input tensor
                )
            }
            for shape in test_configs
        ]

    def generate_sample(self, dtype: torch.dtype = torch.float32) -> List[Dict[str, Any]]:
        """
        Generate a single sample test case for debugging or interactive runs.
        
        Returns:
            A list containing a single test case dictionary
        """
        shape = (4, 4)  # Sample shape (batch_size, num_features)
        return {
            "name": f"shape={shape}",
            "shape": shape,
            "create_inputs": lambda shape=shape: (
                # Create sequential input for easy verification
                torch.randn(*shape, device="cuda", dtype=dtype)
            )
        }

    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the RMS Normalization result is correct.

        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution

        Returns:
            Tuple of (is_correct, debug_info)
        """
        # Use appropriate tolerance based on precision
        rtol = 1e-3 if dtype == torch.float16 else 1e-4
        atol = 1e-3 if dtype == torch.float16 else 1e-5
        
        is_close = torch.allclose(actual_output, expected_output, rtol=rtol, atol=atol)
        
        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            max_diff = torch.max(torch.abs(diff)).item()
            mean_diff = torch.mean(torch.abs(diff)).item()
            
            # Find indices of largest differences (flattening the tensor first)
            flat_diff = torch.abs(diff.flatten())
            _, top_indices_flat = torch.topk(flat_diff, min(5, flat_diff.numel()))
            
            # Convert flat indices back to multi-dimensional indices
            top_indices = []
            shape = expected_output.shape
            for flat_idx in top_indices_flat:
                idx = []
                remaining_idx = flat_idx.item()
                for dim_size in reversed(shape):
                    idx.insert(0, remaining_idx % dim_size)
                    remaining_idx //= dim_size
                top_indices.append(tuple(idx))

            sample_diffs = {
                f"{str(idx)}": {
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
        Get the function signature for the RMS Normalization solution.

        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # X (input)
                ctypes.POINTER(ctypes.c_float),  # Y (output)
                ctypes.c_size_t,                 # batch_size
                ctypes.c_size_t,                 # num_features
            ],
            "restype": None
        }

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the approximate number of floating point operations for RMS Normalization.

        Args:
            test_case: The test case dictionary

        Returns:
            Number of floating point operations
        """
        shape = test_case["shape"]
        batch_size = shape[0]
        num_features = shape[1]
        
        # FLOPs calculation:
        # 1. Calculate the square of each element: num_features multiplications
        # 2. Sum the squared values: (num_features - 1) additions
        # 3. Divide by num_features for mean: 1 division
        # 4. Add epsilon: 1 addition
        # 5. Take the square root: 5 flops (approx)
        # 6. Divide each element by the RMS: num_features divisions
        
        flops_per_batch = (
            num_features +      # Square each element
            (num_features - 1) + # Sum squares
            1 +                 # Divide for mean
            1 +                 # Add epsilon
            5 +                 # Square root (approx cost)
            num_features        # Divide by RMS
        )
        
        # Total operations across the batch
        total_flops = batch_size * flops_per_batch
        
        return int(total_flops)

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.

        Args:
            test_case: The test case dictionary

        Returns:
            List containing batch_size, num_features, total_size, dims_size
        """
        shape = test_case["shape"]
        batch_size = shape[0]
        num_features = shape[1]
        
        return [batch_size, num_features]
