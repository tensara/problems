import torch
import torch.nn as nn
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem 

class frobenius_norm(Problem):
    """Frobenius Normalization problem."""

    is_exact = False

    def __init__(self):
        super().__init__(
            name="frobenius-norm"
        )

    def reference_solution(self, x: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of Frobenius Normalization.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with Frobenius norm normalization applied, same shape as input.
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=x.dtype):
            # Calculate the Frobenius norm
            norm = torch.norm(x, p='fro')
            
            # Normalize the tensor by dividing by the norm
            output = x / norm
            
            return output

    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for Frobenius Normalization.

        Returns:
            List of test case dictionaries with varying sizes
        """
        
        # Define shapes for different test cases
        test_configs = [
            (4, 1024, 1024),       # 3D tensor
            (32, 128, 128),       # 3D tensor
            (8, 32, 256, 256),    # 4D tensor
            (4, 16, 32, 128, 128), # 5D tensor
        ]

        test_cases = []
        for shape in test_configs:
            seed = Problem.get_seed(f"{self.name}_shape={shape}")
            test_cases.append({
                "name": f"shape={shape}",
                "shape": shape,
                "create_inputs": lambda shape=shape, seed=seed, dtype=dtype: (
                    (lambda g: (
                        torch.randn(*shape, device="cuda", dtype=dtype, generator=g), # Input tensor
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
        shape = (4, 4, 4)
        return {
            "name": f"Sample shape={shape}",
            "shape": shape,
            "create_inputs": lambda shape=shape: (
                torch.tensor([
                    [[1.0, 2.0, -1.0, 0.5],
                     [0.0, -2.0, 1.5, -0.5],
                     [2.0, -1.0, 0.0, 1.0],
                     [-0.5, 1.0, -2.0, 0.0]],
                    
                    [[-1.0, 0.5, 2.0, -0.5],
                     [1.0, -1.5, -2.0, 0.5],
                     [-0.5, 2.0, 1.5, -1.0],
                     [0.5, -2.0, 1.0, 0.0]],
                    
                    [[0.5, -1.0, 2.0, -0.5],
                     [-2.0, 1.5, -0.5, 1.0],
                     [1.0, -0.5, 2.0, -1.5],
                     [-1.0, 0.5, -2.0, 1.0]],
                    
                    [[2.0, -0.5, 1.0, -2.0],
                     [-1.0, 0.5, -1.5, 2.0],
                     [0.5, -2.0, 1.0, -0.5],
                     [-1.5, 1.0, -0.5, 2.0]]
                ], device="cuda", dtype=dtype),
            )
        }

    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the Frobenius Normalization result is correct.

        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution

        Returns:
            Tuple of (is_correct, debug_info)
        """        
        is_close = torch.allclose(actual_output, expected_output, rtol=3e-3, atol=1e-6)
        
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
        Get the function signature for the Frobenius Normalization solution.

        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # X (input)
                ctypes.POINTER(ctypes.c_float),  # Y (output)
                ctypes.c_size_t,                 # size (total number of elements)
            ],
            "restype": None
        }

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the approximate number of floating point operations for Frobenius Normalization.

        Args:
            test_case: The test case dictionary

        Returns:
            Number of floating point operations
        """
        shape = test_case["shape"]
        total_elements = 1
        for dim in shape:
            total_elements *= dim
        
        # FLOPs calculation:
        # 1. Calculate the square of each element: N multiplications.
        # 2. Sum all the squared values: N-1 additions.
        # 3. Take the square root: 1 sqrt operation (~5 FLOPs).
        # 4. Divide each element by the norm: N divisions.
        
        # Total FLOPs ≈ N + (N-1) + 5 + N = 3N + 4
        total_flops = 3 * total_elements + 4
        
        return int(total_flops)

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.

        Args:
            test_case: The test case dictionary

        Returns:
            List containing total number of elements
        """
        shape = test_case["shape"]
        total_elements = 1
        for dim in shape:
            total_elements *= dim
        return [total_elements]
