import torch
from typing import List, Dict, Tuple, Any

from problem import Problem

class argmax(Problem):
    """Argmax over dimension problem."""

    is_exact = True

    parameters = [
        {"name": "input", "type": "float", "pointer": True, "const": True},
        {"name": "dim", "type": "int", "pointer": False, "const": False},
        {"name": "output", "type": "int", "pointer": True, "const": False},
        {"name": "shape", "type": "int", "pointer": True, "const": True},
        {"name": "ndim", "type": "int", "pointer": False, "const": False},
    ]

    def __init__(self):
        super().__init__(
            name="argmax"
        )

    def reference_solution(self, input_tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """
        PyTorch implementation of argmax over dimension.

        Args:
            input_tensor: Input tensor of arbitrary shape
            dim: Dimension to perform argmax over

        Returns:
            Result of argmax operation
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=input_tensor.dtype):
            return torch.argmax(input_tensor, dim=dim).to(torch.int32)

    def generate_test_cases(self) -> List[Dict[str, Any]]:
        """
        Generate test cases for argmax over dimension.

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
            name = f"shape={shape}, dim={dim}"
            seed = Problem.get_seed(f"{self.name}_{name}_{(shape, dim)}")
            test_cases.append({
                "name": name,
                "shape": shape,
                "dim": dim,
                "create_inputs": lambda shape=shape, dim=dim, seed=seed, dtype=dtype: (
                    *(lambda g: (
                        torch.rand(shape, device="cuda", dtype=dtype, generator=g) * 10.0 - 5.0,  # uniform [-5, 5]
                    ))(torch.Generator(device="cuda").manual_seed(seed)),
                    dim,
                )
            })
        return test_cases

    def generate_sample(self) -> List[Dict[str, Any]]:
        """
        Generate a single sample test case for debugging or interactive runs.

        Returns:
            A list containing a single test case dictionary
        """
        dtype = self.param_dtype(0)

        shape = (8,8)
        dim = 0
        return {
            "name": f"shape={shape}, dim={dim}",
            "shape": shape,
            "dim": dim,
            "create_inputs": lambda shape=shape, dim=dim: (
                torch.tensor([[1.0, -2.0, 3.0, -4.0, 5.0, -1.5, 2.5, 0.0],
                            [-3.0, 4.0, -1.0, 2.0, -5.0, 1.5, -2.5, 3.5],
                            [2.0, -3.0, 4.0, -2.0, 1.0, -4.0, 3.0, -1.0],
                            [-4.0, 5.0, -3.0, 1.0, -2.0, 4.0, -1.0, 2.0],
                            [3.0, -4.0, 2.0, -1.0, 4.0, -3.0, 1.0, -2.0],
                            [-2.0, 3.0, -4.0, 5.0, -1.0, 2.0, -3.0, 4.0],
                            [4.0, -1.0, 2.0, -3.0, 5.0, -2.0, 3.0, -4.0],
                            [-1.0, 2.0, -3.0, 4.0, -2.0, 3.0, -4.0, 1.0]], device="cuda", dtype=dtype),
                dim
            )
        }

    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the argmax result is correct.

        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution

        Returns:
            Tuple of (is_correct, debug_info)
        """
        is_equal = torch.all(actual_output == expected_output)

        debug_info = {}
        if not is_equal:
            diff_mask = actual_output != expected_output

            # Find indices of differences
            diff_indices = torch.nonzero(diff_mask)
            sample_diffs = {}

            # Get up to 5 sample differences
            for i in range(min(5, len(diff_indices))):
                idx = tuple(diff_indices[i].tolist())
                sample_diffs[f"{i}"] = {
                    "expected": expected_output[idx].item(),
                    "actual": actual_output[idx].item(),
                    "diff": abs(actual_output[idx].item() - expected_output[idx].item())
                }

            debug_info = {
                "max_difference": torch.max(torch.abs(actual_output - expected_output)).item(),
                "sample_differences": sample_diffs
            }

        return is_equal, debug_info

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the number of floating point operations for the problem.

        Args:
            test_case: The test case dictionary

        For argmax, we count:
        - One comparison per element being compared

        Returns:
            Number of floating point operations
        """
        shape = test_case["shape"]
        dim = test_case["dim"]

        # Total elements in the tensor
        total_elements = 1
        for s in shape:
            total_elements *= s

        # Number of elements being compared (size of reduction dimension)
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
            torch.tensor(list(test_case["shape"]), dtype=torch.int32),
            len(test_case["shape"]),
        ]
