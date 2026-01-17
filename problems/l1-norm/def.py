import torch
import torch.nn as nn
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem 

class l1_norm(Problem):
    """L1 Normalization problem."""

    def __init__(self):
        super().__init__(
            name="l1-norm"
        )
        self.epsilon = 1e-10  # Small epsilon for numerical stability

    def reference_solution(self, x: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of L1 Normalization.

        Args:
            x (torch.Tensor): Input tensor of shape (B, D)

        Returns:
            torch.Tensor: Output tensor with L1 Normalization applied, same shape as input.
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            l1_norm = torch.sum(torch.abs(x), dim=1, keepdim=True)
            
            l1_norm = l1_norm + self.epsilon
            
            output = x / l1_norm
            
            return output

    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for L1 Normalization.

        Returns:
            List of test case dictionaries with varying sizes
        """
        
        test_configs = [
            (128, 4096),     # Medium example
            (256, 4096),     # Medium example
            (128, 8192),     # Larger example
            (256, 8192),    # Larger example
            (128, 16384),   # Very large example
        ]

        test_cases = []
        for B, D in test_configs:
            seed = Problem.get_seed(f"{self.name}_B={B}_D={D}")
            test_cases.append({
                "name": f"B={B}, D={D}",
                "B": B,
                "D": D,
                "create_inputs": lambda B=B, D=D, seed=seed, dtype=dtype: (
                    (lambda g: (
                        torch.randn(B, D, device="cuda", dtype=dtype, generator=g), # Input X
                    ))(torch.Generator(device="cuda").manual_seed(seed))
                )
            })
        return test_cases

    def generate_sample(self, dtype: torch.dtype = torch.float32) -> List[Dict[str, Any]]:
        """
        Generate a single sample test case for debugging or interactive runs.
        
        Returns:
            A list containing a single test case dictionary
        """
        B, D = (8, 8)
        return {
            "name": f"Sample B={B}, D={D}",
            "B": B,
            "D": D,
            "create_inputs": lambda B=B, D=D: (
                torch.tensor([[0.1, 0.8, 0.3, 0.9, 0.2, 0.7, 0.4, 0.6],
                                [0.9, 0.2, 0.7, 0.4, 0.1, 0.8, 0.3, 0.5],
                                [0.3, 0.7, 0.1, 0.9, 0.4, 0.6, 0.2, 0.8],
                                [0.8, 0.3, 0.6, 0.1, 0.9, 0.4, 0.7, 0.2],
                                [0.2, 0.9, 0.4, 0.7, 0.1, 0.8, 0.3, 0.6],
                                [0.7, 0.2, 0.8, 0.3, 0.6, 0.1, 0.9, 0.4],
                                [0.4, 0.8, 0.2, 0.6, 0.3, 0.9, 0.1, 0.7],
                                [0.6, 0.1, 0.9, 0.4, 0.7, 0.2, 0.8, 0.3]], device="cuda", dtype=dtype),
            )
        }

    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the L1 Normalization result is correct.

        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution

        Returns:
            Tuple of (is_correct, debug_info)
        """
        # Use appropriate tolerance based on dtype
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
        Get the function signature for the L1 Normalization solution.

        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            # Corresponds to parameters in problem.md
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # X (input)
                ctypes.POINTER(ctypes.c_float),  # Y (output)
                ctypes.c_size_t,                 # B (batch size)
                ctypes.c_size_t,                 # D (dimension)
            ],
            "restype": None
        }

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the approximate number of floating point operations for L1 Normalization.

        Args:
            test_case: The test case dictionary

        IMPORTANT: Comments are required. Outline the FLOPs calculation.

        Returns:
            Number of floating point operations
        """
        B = test_case["B"]
        D = test_case["D"]

        # FLOPs calculation per batch item:
        # 1. Calculate absolute values: D abs operations. ~D FLOPs.
        # 2. Calculate sum: (D-1) adds. ~D FLOPs.
        # 3. Add epsilon: 1 addition. ~1 FLOPs.
        # 4. Division: D divisions. ~D FLOPs.
        
        # Total FLOPs per batch item ≈ D + D + 1 + D = 3D + 1
        flops_per_batch_item = 3 * D + 1
        
        # Total FLOPs for the batch
        total_flops = B * flops_per_batch_item
        
        return int(total_flops) # Return as integer

    def get_mem(self, test_case: Dict[str, Any]) -> int:
        """
        Get the memory usage for the problem. Assumed to be all in DRAM
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            Memory usage in bytes
        """
        B = test_case["B"]
        D = test_case["D"]
        
        # Input: B*D elements, Output: B*D elements (same shape)
        dtype_bytes = 4  # 4 bytes per float32 element
        return (B * D + B * D) * dtype_bytes

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters (dimensions) to pass to the CUDA solution.

        Args:
            test_case: The test case dictionary

        Returns:
            List containing B, D
        """
        B = test_case["B"]
        D = test_case["D"]
        return [B, D]

