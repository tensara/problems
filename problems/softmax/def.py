import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class softmax(Problem):
    """Softmax function problem."""

    def __init__(self):
        super().__init__(
            name="softmax"
        )

    def reference_solution(self, input_tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """
        PyTorch implementation of softmax function.

        Args:
            input_tensor: Input tensor of arbitrary shape
            dim: Dimension to compute softmax over

        Returns:
            Softmax probabilities along the specified dimension
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            return torch.nn.functional.softmax(input_tensor, dim=dim)

    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for softmax function.

        Returns:
            List of test case dictionaries with varying sizes and dimensions
        """
        test_configs = [
            ((16, 128, 256), 1, "normal"),      # Typical hidden layer
            ((32, 512, 512), 2, "uniform"),     # Large feature dimension
            ((8, 1024, 1024), 1, "normal"),     # Very large sequence length
            ((64, 128, 128, 128), 2, "uniform"),# 4D tensor
            ((4, 256, 256, 256), 3, "normal"),  # Large 4D tensor
            ((128, 10), 1, "normal"),           # Classification logits
            ((256, 50, 50), 0, "uniform")       # First dimension softmax
        ]

        test_cases = []
        for shape, dim, dist in test_configs:
            name = f"shape={shape}, dim={dim}, dist={dist}"
            seed = Problem.get_seed(f"{self.name}_{name}_{(shape, dim, dist)}")
            test_cases.append({
                "name": name,
                "shape": shape,
                "dim": dim,
                "seed": seed,
                "create_inputs": lambda shape=shape, dim=dim, seed=seed, dist=dist, dtype=dtype: (
                    (lambda g: (
                        torch.randn(shape, device="cuda", dtype=dtype, generator=g) * 2.0
                        if dist == "normal" else
                        (torch.rand(shape, device="cuda", dtype=dtype, generator=g) - 0.5) * 6.0
                    ))(torch.Generator(device="cuda").manual_seed(seed)),
                    dim,
                )
            })
        return test_cases

    def generate_sample(self, dtype: torch.dtype = torch.float32) -> List[Dict[str, Any]]:
        """
        Generate a single sample test case for debugging or interactive runs.

        Returns:
            A list containing a single test case dictionary
        """
        shape = (4, 4)
        dim = 1
        return {
            "name": f"shape={shape}, dim={dim}",
            "shape": shape,
            "dim": dim,
            "create_inputs": lambda shape=shape, dim=dim: (
                # Create predictable logits for easy verification
                torch.randn(shape, device="cuda", dtype=dtype) * 2.0,
                dim
            )
        }

    def verify_result(self, expected_output: torch.Tensor,
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the softmax result is correct.

        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution

        Returns:
            Tuple of (is_correct, debug_info)
        """
        # Check if outputs are close
        is_close = torch.allclose(actual_output, expected_output, rtol=1e-3, atol=1e-3)

        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            max_diff = torch.max(torch.abs(diff)).item()
            mean_diff = torch.mean(torch.abs(diff)).item()

            # Find indices of largest differences
            flat_diff = diff.flatten()
            _, top_indices = torch.topk(torch.abs(flat_diff), min(5, flat_diff.numel()))

            # Get sample differences
            sample_diffs = {}
            for i, idx in enumerate(top_indices):
                sample_diffs[f"{i}"] = {
                    "expected": expected_output.flatten()[idx].item(),
                    "actual": actual_output.flatten()[idx].item(),
                    "diff": diff.flatten()[idx].item()
                }

            debug_info = {
                "max_difference": max_diff,
                "mean_difference": mean_diff,
                "sample_differences": sample_diffs
            }

        return is_close, debug_info

    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the softmax solution.

        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # input_tensor
                ctypes.c_int,                    # dim
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.POINTER(ctypes.c_size_t), # shape
                ctypes.c_size_t,                 # ndim
            ],
            "restype": None
        }

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the number of floating point operations for the problem.

        Args:
            test_case: The test case dictionary

        For softmax, we count:
        - One max operation per slice
        - One subtraction per element
        - One exp per element
        - One sum per slice
        - One division per element

        Returns:
            Number of floating point operations
        """
        shape = test_case["shape"]
        dim = test_case["dim"]

        # Total elements in the tensor
        total_elements = 1
        for s in shape:
            total_elements *= s

        # Size of the dimension we're computing softmax over
        dim_size = shape[dim]

        # Number of slices
        num_slices = total_elements // dim_size

        # FLOPs per slice:
        # - dim_size-1 comparisons for max
        # - dim_size subtractions
        # - dim_size exponentials
        # - dim_size-1 additions for sum
        # - dim_size divisions
        flops_per_slice = (
            (dim_size - 1) +  # max
            dim_size +        # subtract
            dim_size +        # exp
            (dim_size - 1) +  # sum
            dim_size          # divide
        )

        return num_slices * flops_per_slice

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.

        Args:
            test_case: The test case dictionary

        Returns:
            List containing the shape array and number of dimensions
        """
        return [
            torch.tensor(list(test_case["shape"]), dtype=torch.int64, device="cuda"),
            len(test_case["shape"]),
        ]
