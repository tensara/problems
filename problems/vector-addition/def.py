import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem

class vector_addition(Problem):
    """Vector Addition problem."""

    def __init__(self):
        super().__init__(
            name="vector-addition"
        )

    def reference_solution(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of vector addition.

        Args:
            a: First input tensor (vector)
            b: Second input tensor (vector)

        Returns:
            Result of a + b
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            return a + b

    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for vector addition.

        Returns:
            List of test case dictionaries with varying sizes
        """
        test_configs = [
            # (size)
            (128,),
            (256,),
            (512,),
            (1024,),
            (2048,),
            (4096,),
            (16 * 1024,),
            (128 * 1024,),
            (1024 * 1024,) # 1M elements
        ]

        return [
            {
                "name": f"size={config[0]}",
                "size": config[0],
                "create_inputs": lambda size=config[0]: (
                    torch.rand(size, device="cuda", dtype=dtype) * 10.0 - 5.0,  # uniform [-5, 5]
                    torch.rand(size, device="cuda", dtype=dtype) * 10.0 - 5.0,
                )
            }
            for config in test_configs
        ]

    def generate_sample(self, dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
        """
        Generate a single sample test case for debugging or interactive runs.

        Returns:
            A test case dictionary
        """
        size = 8
        return {
            "name": f"size={size}",
            "size": size,
            "create_inputs": lambda size=size: (
                torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device="cuda", dtype=dtype),
                torch.tensor([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], device="cuda", dtype=dtype),
            )
        }

    def verify_result(self, expected_output: torch.Tensor,
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the vector addition result is correct.

        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            dtype: The data type of the tensors

        Returns:
            Tuple of (is_correct, debug_info)
        """
        # For floating point, use allclose for comparison
        is_correct = torch.allclose(actual_output, expected_output, atol=1e-5, rtol=1e-4)

        debug_info = {}
        if not is_correct:
            diff = torch.abs(actual_output - expected_output)
            max_diff = torch.max(diff).item()
            
            # Find indices of differences
            diff_mask = ~torch.isclose(actual_output, expected_output, atol=1e-5, rtol=1e-4)
            diff_indices = torch.nonzero(diff_mask)
            sample_diffs = {}
            
            # Get up to 5 sample differences
            for i in range(min(5, len(diff_indices))):
                idx = tuple(diff_indices[i].tolist()) # Assumes 1D tensor for simplicity
                sample_diffs[f"index_{idx[0]}"] = { # Access first element of tuple for 1D index
                    "expected": expected_output[idx].item(),
                    "actual": actual_output[idx].item(),
                    "difference": diff[idx].item()
                }

            debug_info = {
                "max_absolute_difference": max_diff,
                "sample_differences": sample_diffs,
                "mismatched_elements": diff_indices.shape[0] if diff_indices.numel() > 0 else 0,
                "total_elements": expected_output.numel()
            }

        return is_correct, debug_info

    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the vector addition solution.

        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # input_tensor_a
                ctypes.POINTER(ctypes.c_float),  # input_tensor_b
                ctypes.POINTER(ctypes.c_float),  # output_tensor
                ctypes.c_int32,                    # size (N)
            ],
            "restype": None
        }

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the number of floating point operations for the problem.

        Args:
            test_case: The test case dictionary

        For vector addition of size N, there are N additions.
        Returns:
            Number of floating point operations
        """
        return test_case["size"] # N additions

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.

        Args:
            test_case: The test case dictionary

        Returns:
            List containing the size of the vectors
        """
        return [
            test_case["size"],
        ]