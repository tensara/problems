import torch
import torch.nn as nn
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem 

class batch_norm(Problem):
    """Batch Normalization problem."""

    is_exact = False

    def __init__(self):
        super().__init__(
            name="batch-norm"
        )
        self.epsilon = 1e-5  # Standard epsilon for BatchNorm

    def reference_solution(self, x: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of Batch Normalization using nn.BatchNorm2d.

        Args:
            x (torch.Tensor): Input tensor of shape (B, F, D1, D2)

        Returns:
            torch.Tensor: Output tensor with Batch Normalization applied, same shape as input.
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=x.dtype):
            # Create BatchNorm2d layer with no affine parameters and no running stats
            bn = nn.BatchNorm2d(
                num_features=x.size(1),  # F dimension
                affine=False,  # No learnable parameters
                track_running_stats=False,  # Don't track running stats
                eps=self.epsilon
            )
            return bn(x)

    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for Batch Normalization.

        Returns:
            List of test case dictionaries with varying sizes
        """
        
        # Define shapes: (B, F, D1, D2)
        test_configs = [
            (16, 64, 256, 256),  # Base example from the model
            (32, 128, 128, 128), # Medium example
            (8, 256, 64, 64),    # Larger channels, smaller spatial
            (4, 32, 512, 512),   # Small batch, large spatial
        ]

        test_cases = []
        for B, F, D1, D2 in test_configs:
            name = f"B={B}, F={F}, D1={D1}, D2={D2}"
            seed = Problem.get_seed(f"{self.name}_{name}_{(B, F, D1, D2)}")
            test_cases.append({
                "name": name,
                "B": B,
                "F": F,
                "D1": D1,
                "D2": D2,
                "create_inputs": lambda B=B, F=F, D1=D1, D2=D2, seed=seed, dtype=dtype: (
                    *(lambda g: (
                        torch.randn(B, F, D1, D2, device="cuda", dtype=dtype, generator=g), # Input X
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
        B, F, D1, D2 = (2, 2, 2, 2) # Sample configuration
        return {
            "name": f"B={B}, F={F}, D1={D1}, D2={D2}",
            "B": B,
            "F": F,
            "D1": D1,
            "D2": D2,
            "create_inputs": lambda B=B, F=F, D1=D1, D2=D2: (
                torch.tensor([
                    [[[1.0, -2.0], [0.5, -1.5]], [[-0.5, 2.0], [1.5, -1.0]]],
                    [[[-1.0, 0.5], [2.0, -0.5]], [[1.0, -1.5], [-2.0, 0.5]]]
                ], device="cuda", dtype=dtype),
            )
        }

    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the Batch Normalization result is correct.

        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution

        Returns:
            Tuple of (is_correct, debug_info)
        """
        # Use a slightly higher tolerance for BatchNorm due to potential precision differences
        rtol = 1e-3 if dtype == torch.float16 else 1e-4
        atol = 1e-3 if dtype == torch.float16 else 1e-5
        
        is_close = torch.allclose(actual_output, expected_output, rtol=rtol, atol=atol)
        
        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            max_diff = torch.max(torch.abs(diff)).item()
            mean_diff = torch.mean(torch.abs(diff)).item()
            
            # Find indices of largest differences
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
        Get the function signature for the Batch Normalization solution.

        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            # Corresponds to parameters in problem.md
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # X (input)
                ctypes.POINTER(ctypes.c_float),  # Y (output)
                ctypes.c_size_t,                 # B (batch size)
                ctypes.c_size_t,                 # F (features)
                ctypes.c_size_t,                 # D1 (dim1)
                ctypes.c_size_t                  # D2 (dim2)
            ],
            "restype": None
        }

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the approximate number of floating point operations for Batch Normalization.

        Args:
            test_case: The test case dictionary

        Returns:
            Number of floating point operations
        """
        B = test_case["B"]
        F = test_case["F"]
        D1 = test_case["D1"]
        D2 = test_case["D2"]
        N = B  # Number of elements to normalize per feature location

        # FLOPs calculation per feature location (F * D1 * D2 locations):
        # 1. Calculate mean: Sum B elements (B-1 adds), 1 division. ~B FLOPs.
        # 2. Calculate variance: (x - mean)^2 (B subtractions, B squares), sum B squares (B-1 adds), 1 division. ~3B FLOPs.
        # 3. Normalize: x - mean (B subtractions), sqrt(var + eps) (1 addition, 1 sqrt), division (B divisions). ~2B + sqrt_cost FLOPs.
        
        # Total FLOPs per feature location ≈ B + 3B + 2B + 2 = 6B + 2
        flops_per_location = 6 * B + 2
        
        # Total FLOPs for all feature locations
        total_flops = flops_per_location * F * D1 * D2
        
        return int(total_flops)

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters (dimensions) to pass to the CUDA solution.

        Args:
            test_case: The test case dictionary

        Returns:
            List containing B, F, D1, D2
        """
        B = test_case["B"]
        F = test_case["F"]
        D1 = test_case["D1"]
        D2 = test_case["D2"]
        return [B, F, D1, D2]
