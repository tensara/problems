import torch
from typing import List, Dict, Tuple, Any

from problem import Problem

class max_dim(Problem):
    """Max over dimension problem."""

    is_exact = True

    parameters = [
        {"name": "input", "type": "float", "pointer": True, "const": True},
        {"name": "dim", "type": "int", "pointer": False, "const": False},
        {"name": "output", "type": "float", "pointer": True, "const": False},
        {"name": "shape", "type": "size_t", "pointer": True, "const": True},
        {"name": "ndim", "type": "size_t", "pointer": False, "const": False},
    ]

    def __init__(self):
        super().__init__(
            name="max-dim"
        )

    def reference_solution(self, input_tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """
        PyTorch implementation of max over dimension.

        Args:
            input_tensor: Input tensor of arbitrary shape
            dim: Dimension to reduce over

        Returns:
            Result of max reduction with keepdim=True (values only)
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=input_tensor.dtype):
            # Get only the values, not the indices
            return torch.max(input_tensor, dim=dim, keepdim=True)[0]

    def generate_test_cases(self) -> List[Dict[str, Any]]:
        """
        Generate test cases for max over dimension.

        Returns:
            List of test case dictionaries with varying sizes and dimensions
        """
        dtype = self.param_dtype(0)

        test_configs = [
            # (shape, dim)
            ((16, 128, 256), 1),
            ((32, 512, 512), 0),
            ((8, 1024, 1024), 2),
            ((64, 128, 128, 128), 2),
            ((4, 256, 256, 256), 1),
            ((128, 64, 64, 64), 3)
        ]

        test_cases = []
        for shape, dim in test_configs:
            seed = Problem.get_seed(f"{self.name}_shape={shape}_dim={dim}")
            test_cases.append({
                "name": f"shape={shape}, dim={dim}",
                "shape": shape,
                "dim": dim,
                "create_inputs": lambda shape=shape, dim=dim, seed=seed, dtype=dtype: (
                    *(lambda g: (
                        torch.rand(shape, device="cuda", dtype=dtype, generator=g) * 10.0 - 5.0,  # uniform [-5, 5]
                    ))(torch.Generator(device="cuda").manual_seed(seed)),
                    dim
                )
            })
        return test_cases

    def generate_sample(self) -> List[Dict[str, Any]]:
        """
        Generate sample test cases for max over dimension with predictable inputs.

        Returns:
            List of sample test case dictionaries.
        """
        dtype = self.param_dtype(0)

        shape = (4, 4, 4)
        dim = 1
        return {
            "name": f"shape={shape}, dim={dim}",
            "shape": shape,
            "dim": dim,
            "create_inputs": lambda shape=shape, dim=dim: (
                torch.rand(shape, device="cuda", dtype=dtype) * 10.0 - 5.0,
                dim
            )
        }

    def verify_result(self, expected_output: torch.Tensor,
                     actual_output: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the max reduction result is correct.

        Args:
            expected_output: Output from reference solution (max values)
            actual_output: Output from submitted solution (max values)

        Returns:
            Tuple of (is_correct, debug_info)
        """
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

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the number of floating point operations for the problem.

        Args:
            test_case: The test case dictionary

        For max reduction, we count:
        - One comparison per element being reduced

        Returns:
            Number of floating point operations
        """
        shape = test_case["shape"]
        dim = test_case["dim"]

        # Total elements in the tensor
        total_elements = 1
        for s in shape:
            total_elements *= s

        # Number of elements being reduced (size of reduction dimension)
        reduce_size = shape[dim]

        # Number of reduction operations
        num_reductions = total_elements // reduce_size

        # Each reduction requires (reduce_size - 1) comparisons
        return num_reductions * (reduce_size - 1)

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
