import torch
import torch.nn as nn
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem 

class layer_norm(Problem):
    """Layer Normalization problem."""

    def __init__(self):
        super().__init__(
            name="layer-norm"
        )
        self.epsilon = 1e-5 # Standard epsilon for LayerNorm

    def reference_solution(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of Layer Normalization.

        Args:
            x (torch.Tensor): Input tensor of shape (B, F, D1, D2)
            gamma (torch.Tensor): Scale tensor of shape (F, D1, D2)
            beta (torch.Tensor): Shift tensor of shape (F, D1, D2)

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            # Normalize over the last 3 dimensions (F, D1, D2)
            normalized_shape = x.shape[1:] 
            
            # Use torch.nn.functional.layer_norm
            output = torch.nn.functional.layer_norm(
                x, 
                normalized_shape, 
                weight=gamma, 
                bias=beta, 
                eps=self.epsilon
            )
            return output

    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for Layer Normalization.

        Returns:
            List of test case dictionaries with varying sizes
        """
        
        # Define shapes: (B, F, D1, D2)
        test_configs = [
            (16, 64, 32, 32),   # Smaller example
            (32, 128, 64, 64),  # Medium example
            (8, 256, 128, 128), # Larger example
            (4, 512, 32, 32),   # Different aspect ratio
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
                    torch.randn(F, D1, D2, device="cuda", dtype=dtype),      # Gamma (scale)
                    torch.randn(F, D1, D2, device="cuda", dtype=dtype)       # Beta (shift)
                )
            }
            for B, F, D1, D2 in test_configs
        ]

    def generate_sample(self, dtype: torch.dtype = torch.float32) -> List[Dict[str, Any]]:
        """
        Generate a single sample test case for debugging or interactive runs.
        
        Returns:
            A list containing a single test case dictionary
        """
        B, F, D1, D2 = (2, 4, 4, 4)
        return {
            "name": f"Sample B={B}, F={F}, D1={D1}, D2={D2}",
            "B": B,
            "F": F,
            "D1": D1,
            "D2": D2,
            "create_inputs": lambda B=B, F=F, D1=D1, D2=D2: (
                torch.tensor([
                    [[[0, 0.5, 1, 1.5], [2, 2.5, 3, 3.5], [4, 4.5, 5, 5.5], [6, 6.5, 7, 7.5]],
                        [[-1, -1.5, -2, -2.5], [-3, -3.5, -4, -4.5], [-5, -5.5, -6, -6.5], [-7, -7.5, -8, -8.5]],
                        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
                        [[-2, -3, -4, -5], [-6, -7, -8, -9], [-10, -11, -12, -13], [-14, -15, -16, -17]]
                    ],
                    [[[0.5, 1, 1.5, 2], [2.5, 3, 3.5, 4], [4.5, 5, 5.5, 6], [6.5, 7, 7.5, 8]],
                        [[-1.5, -2, -2.5, -3], [-3.5, -4, -4.5, -5], [-5.5, -6, -6.5, -7], [-7.5, -8, -8.5, -9]],
                        [[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13], [14, 15, 16, 17]],
                        [[-3, -4, -5, -6], [-7, -8, -9, -10], [-11, -12, -13, -14], [-15, -16, -17, -18]]
                    ]
                ], device="cuda", dtype=dtype),
                torch.ones(F, D1, D2, device="cuda", dtype=dtype), # Gamma = 1
                torch.zeros(F, D1, D2, device="cuda", dtype=dtype)  # Beta = 0
            )
        }

    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the Layer Normalization result is correct.

        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution

        Returns:
            Tuple of (is_correct, debug_info)
        """
        # Use a slightly higher tolerance for LayerNorm due to potential precision differences
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
        Get the function signature for the Layer Normalization solution.

        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            # Corresponds to parameters in problem.md
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # X (input)
                ctypes.POINTER(ctypes.c_float),  # gamma (scale)
                ctypes.POINTER(ctypes.c_float),  # beta (shift)
                ctypes.POINTER(ctypes.c_float),  # Y (output)
                ctypes.c_size_t,                 # B (batch size)
                ctypes.c_size_t,                 # F (features)
                ctypes.c_size_t,                 # D1 (dim1)
                ctypes.c_size_t                  # D2 (dim2)
                # Epsilon is handled internally or passed differently if needed by CUDA kernel
            ],
            "restype": None
        }

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the approximate number of floating point operations for Layer Normalization.

        Args:
            test_case: The test case dictionary

        IMPORTANT: Comments are required. Outline the FLOPs calculation.

        Returns:
            Number of floating point operations
        """
        B = test_case["B"]
        F = test_case["F"]
        D1 = test_case["D1"]
        D2 = test_case["D2"]
        N = F * D1 * D2 # Number of elements to normalize per batch item

        # FLOPs calculation per batch item:
        # 1. Calculate mean: Sum N elements (N-1 adds), 1 division. ~N FLOPs.
        # 2. Calculate variance: (x - mean)^2 (N subtractions, N squares/multiplications), sum N squares (N-1 adds), 1 division. ~3N FLOPs.
        # 3. Normalize: x - mean (N subtractions), sqrt(var + eps) (N additions, N sqrt ops), division (N divisions). ~3N + N*sqrt_cost FLOPs. Let's approx sqrt cost as ~5 FLOPs. ~8N FLOPs.
        # 4. Scale and shift: y * gamma (N multiplications), + beta (N additions). ~2N FLOPs.
        
        # Total FLOPs per batch item ≈ N + 3N + 8N + 2N = 14N
        flops_per_batch_item = 14 * N
        
        # Total FLOPs for the batch
        total_flops = B * flops_per_batch_item
        
        return int(total_flops) # Return as integer

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
