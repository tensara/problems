import torch
import torch.nn as nn
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem 

class instance_norm(Problem):
    """Instance Normalization problem."""

    def __init__(self):
        super().__init__(
            name="instance-norm"
        )
        self.epsilon = 1e-5  # Standard epsilon for InstanceNorm

    def reference_solution(self, x: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of Instance Normalization (2D only).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width)

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            # Get the number of features (channels)
            num_features = x.shape[1]
            
            # Use PyTorch's built-in InstanceNorm2d
            inorm = nn.InstanceNorm2d(num_features=num_features, eps=self.epsilon)
            
            # Move to the same device as the input
            inorm = inorm.to(x.device)
            
            # Apply instance normalization
            output = inorm(x)
            
            return output

    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for Instance Normalization.

        Returns:
            List of test case dictionaries with varying sizes
        """
        
        # Define configurations: (batch_size, features, height, width)
        test_configs = [
            (16, 64, 32, 32),    # 4D tensor, moderate size
            (32, 128, 64, 64),   # 4D tensor, larger size
            (8, 32, 16, 16),     # 4D tensor, smaller size
            (4, 16, 128, 128),   # 4D tensor, high resolution
            (64, 3, 224, 224)    # 4D tensor, typical image batch
        ]

        return [
            {
                "name": f"batch={batch}, features={features}, height={height}, width={width}",
                "batch": batch,
                "features": features, 
                "height": height,
                "width": width,
                "create_inputs": lambda batch=batch, features=features, height=height, width=width: (
                    torch.randn(batch, features, height, width, device="cuda", dtype=dtype),  # Input tensor
                )
            }
            for batch, features, height, width in test_configs
        ]

    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the Instance Normalization result is correct.

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
        Get the function signature for the Instance Normalization solution.

        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # X (input)
                ctypes.POINTER(ctypes.c_float),  # Y (output)
                ctypes.c_size_t,                 # batch_size
                ctypes.c_size_t,                 # num_features (channels)
                ctypes.c_size_t,                 # height
                ctypes.c_size_t,                 # width
            ],
            "restype": None
        }

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the approximate number of floating point operations for Instance Normalization.

        Args:
            test_case: The test case dictionary

        Returns:
            Number of floating point operations
        """
        batch = test_case["batch"]
        features = test_case["features"]
        height = test_case["height"]
        width = test_case["width"]
        
        # Calculate spatial size
        spatial_size = height * width
        
        # Elements per channel instance
        elements_per_instance = spatial_size
        
        # FLOPs calculation per channel instance:
        # For each of the batch * features instances:
        # 1. Calculate mean: spatial_size additions, 1 division
        # 2. Calculate variance: spatial_size subtractions, spatial_size multiplications,
        #    spatial_size additions, 1 division
        # 3. Normalize: spatial_size subtractions, 1 sqrt, spatial_size divisions
        
        # Per instance operations
        flops_per_instance = (
            elements_per_instance + 1 +                  # Mean calculation
            3 * elements_per_instance + 1 +              # Variance calculation
            elements_per_instance + 5 + elements_per_instance  # Normalization (5 for sqrt)
        )
        
        # Total operations across all instances and the batch
        total_flops = batch * features * flops_per_instance
        
        return int(total_flops)

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.

        Args:
            test_case: The test case dictionary

        Returns:
            List containing batch_size, num_features, height, width
        """
        batch = test_case["batch"]
        features = test_case["features"]
        height = test_case["height"]
        width = test_case["width"]
            
        return [batch, features, height, width]
