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
            (16, 64),                   # 2D tensor (batch, features)
            (32, 128),                  # 2D tensor, larger size
            (8, 32, 16, 16),            # 4D tensor (batch, features, height, width)
            (4, 64, 32, 32),            # 4D tensor, larger size
            (64, 768)                   # 2D tensor, typical for transformer models
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
                ctypes.c_size_t,                 # total_size (total number of elements)
                ctypes.c_size_t,                 # dims_size (number of elements per feature)
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
        
        # Calculate number of elements per feature
        dims_size = 1
        for i in range(2, len(shape)):
            dims_size *= shape[i]
        
        # Total elements per batch
        elements_per_batch = num_features * dims_size
        
        # FLOPs calculation:
        # 1. Calculate the square of each element: elements_per_batch multiplications
        # 2. Sum the squared values per feature: (elements_per_batch - dims_size) additions
        # 3. Divide by dims_size for mean: dims_size divisions
        # 4. Add epsilon: dims_size additions
        # 5. Take the square root: dims_size sqrt operations (approx. 5 flops each)
        # 6. Divide each element by the RMS: elements_per_batch divisions
        
        flops_per_batch = (
            elements_per_batch +                 # Square each element
            (num_features * (dims_size - 1)) +   # Sum squares per feature
            num_features +                       # Divide by dims_size for mean
            num_features +                       # Add epsilon
            (5 * num_features) +                 # Square root (approx cost)
            elements_per_batch                   # Divide by RMS
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
        
        # Calculate total size and dims_size
        total_size = 1
        for dim in shape:
            total_size *= dim
            
        # Calculate size of dimensions after batch and features
        dims_size = 1
        for i in range(2, len(shape)):
            dims_size *= shape[i]
        
        # If there are no extra dimensions (2D tensor), dims_size is 1
        if len(shape) <= 2:
            dims_size = 1
            
        return [batch_size, num_features, total_size, dims_size]
