import torch
import torch.nn as nn
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem
from tinygrad.tensor import Tensor
from .solution import BatchNormSolutions

class batch_norm(Problem, BatchNormSolutions):
    """Batch Normalization problem."""

    def __init__(self):
        super().__init__(
            name="batch-norm"
        )
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

        return [
            {
                "name": f"B={B}, F={F}, D1={D1}, D2={D2}",
                "B": B,
                "F": F,
                "D1": D1,
                "D2": D2,
                "create_inputs": lambda B=B, F=F, D1=D1, D2=D2: (
                    torch.randn(B, F, D1, D2, device="cuda", dtype=dtype), # Input X
                )
            }
            for B, F, D1, D2 in test_configs
        ]

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
