import torch
import torch.nn as nn
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem 

class group_norm(Problem):
    """Group Normalization problem."""

    def __init__(self):
        super().__init__(
            name="group-norm"
        )
        self.epsilon = 1e-5  # Standard epsilon for GroupNorm

    def reference_solution(self, x: torch.Tensor, num_groups: int) -> torch.Tensor:
        """
        PyTorch implementation of Group Normalization.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *)
            num_groups (int): Number of groups to divide the channels into

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            # Get the number of features (channels)
            num_features = x.shape[1]
            
            # Use PyTorch's built-in GroupNorm
            gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_features, eps=self.epsilon)
            
            # Move to the same device as the input
            gn = gn.to(x.device)
            
            # Apply group normalization
            output = gn(x)
            
            return output

    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for Group Normalization.

        Returns:
            List of test case dictionaries with varying sizes
        """
        
        # Define configurations: (batch_size, features, dims..., num_groups)
        test_configs = [
            (16, 64, 32, 32, 8),               # 4D tensor, moderate size
            (32, 128, 64, 64, 16),             # 4D tensor, larger size
            (8, 32, 16, 16, 4),                # 4D tensor, smaller size
            (4, 64, 32, 32, 32, 8),            # 5D tensor
            (16, 32, 8)                        # 3D tensor (batch, features, spatial)
        ]

        return [
            {
                "name": f"batch={batch}, features={features}, groups={num_groups}, dims={dims}",
                "batch": batch,
                "features": features, 
                "dims": dims,
                "num_groups": num_groups,
                "create_inputs": lambda batch=batch, features=features, dims=dims, num_groups=num_groups: (
                    torch.randn(batch, features, *dims, device="cuda", dtype=dtype),  # Input tensor
                    num_groups                                                        # Number of groups
                )
            }
            for batch, features, *dims, num_groups in test_configs
        ]

    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the Group Normalization result is correct.

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
        Get the function signature for the Group Normalization solution.

        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # X (input)
                ctypes.POINTER(ctypes.c_float),  # Y (output)
                ctypes.c_size_t,                 # batch_size
                ctypes.c_size_t,                 # num_features (channels)
                ctypes.c_size_t,                 # num_groups
                ctypes.c_size_t,                 # spatial_size (flattened spatial dimensions)
            ],
            "restype": None
        }

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the approximate number of floating point operations for Group Normalization.

        Args:
            test_case: The test case dictionary

        Returns:
            Number of floating point operations
        """
        batch = test_case["batch"]
        features = test_case["features"]
        dims = test_case["dims"]
        num_groups = test_case["num_groups"]
        
        # Calculate spatial size
        spatial_size = 1
        for dim in dims:
            spatial_size *= dim
        
        # Elements per group
        channels_per_group = features // num_groups
        elements_per_group = channels_per_group * spatial_size
        
        # FLOPs calculation per group:
        # For each of the batch * num_groups groups:
        # 1. Calculate mean: elements_per_group additions, 1 division
        # 2. Calculate variance: elements_per_group subtractions, elements_per_group multiplications,
        #    elements_per_group additions, 1 division
        # 3. Normalize: elements_per_group subtractions, 1 sqrt, elements_per_group divisions
        
        # Per group operations
        flops_per_group = (
            elements_per_group + 1 +                  # Mean calculation
            3 * elements_per_group + 1 +              # Variance calculation
            elements_per_group + 5 + elements_per_group  # Normalization (5 for sqrt)
        )
        
        # Total operations across all groups and the batch
        total_flops = batch * num_groups * flops_per_group
        
        # Plus the per-element scale and shift (2 operations per element)
        total_elements = batch * features * spatial_size
        total_flops += 2 * total_elements
        
        return int(total_flops)

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.

        Args:
            test_case: The test case dictionary

        Returns:
            List containing batch_size, num_features, num_groups, spatial_size
        """
        batch = test_case["batch"]
        features = test_case["features"]
        dims = test_case["dims"]
        num_groups = test_case["num_groups"]
        
        # Calculate spatial size (product of all spatial dimensions)
        spatial_size = 1
        for dim in dims:
            spatial_size *= dim
            
        return [batch, features, num_groups, spatial_size]
