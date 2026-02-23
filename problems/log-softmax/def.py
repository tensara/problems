import torch
from typing import List, Dict, Tuple, Any

from problem import Problem


class log_softmax(Problem):
    """Log-Softmax activation function problem."""

    is_exact = False

    parameters = [
        {"name": "input", "type": "float", "pointer": True, "const": True},
        {"name": "output", "type": "float", "pointer": True, "const": False},
        {"name": "M", "type": "size_t", "pointer": False, "const": False},
        {"name": "N", "type": "size_t", "pointer": False, "const": False},
    ]

    def __init__(self):
        super().__init__(name="log-softmax")

    def reference_solution(self, input_matrix: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of Log-Softmax applied row-wise.

        Args:
            input_matrix: Input matrix of shape (M, N)

        Returns:
            Result of log-softmax activation applied along dim=1
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=input_matrix.dtype):
            return torch.nn.functional.log_softmax(input_matrix, dim=1)

    def generate_test_cases(self) -> List[Dict[str, Any]]:
        """
        Generate test cases for Log-Softmax.

        Returns:
            List of test case dictionaries with varying sizes
        """
        dtype = self.param_dtype(0)

        test_configs = [
            ("4096x4096", 4096, 4096),
            ("6144x4096", 6144, 4096),
            ("4096x7168", 4096, 7168),
            ("4096x8192", 4096, 8192),
            ("8192x8192", 8192, 8192),
        ]

        test_cases = []
        for name, m, n in test_configs:
            seed = Problem.get_seed(f"{self.name}_{name}_{(m, n)}")
            test_cases.append({
                "name": name,
                "rows": m,
                "cols": n,
                "create_inputs": lambda m=m, n=n, seed=seed, dtype=dtype: (
                    *(lambda g: (
                        torch.rand((m, n), device="cuda", dtype=dtype, generator=g) * 2.0 - 1.0,
                    ))(torch.Generator(device="cuda").manual_seed(seed)),
                )
            })
        return test_cases

    def generate_sample(self) -> Dict[str, Any]:
        """
        Generate a single sample test case for debugging or interactive runs.

        Returns:
            A dictionary containing a single test case
        """
        dtype = self.param_dtype(0)

        m, n = (4, 8)
        return {
            "name": f"Sample ({m}x{n})",
            "rows": m,
            "cols": n,
            "create_inputs": lambda m=m, n=n: (
                torch.linspace(-3, 3, m * n, device="cuda", dtype=dtype).view(m, n),
            )
        }

    def verify_result(
        self,
        expected_output: torch.Tensor,
        actual_output: torch.Tensor,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the log-softmax result is correct.

        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution

        Returns:
            Tuple of (is_correct, debug_info)
        """
        is_close = torch.allclose(actual_output, expected_output, rtol=1e-4, atol=2e-5)

        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            max_diff = torch.max(torch.abs(diff)).item()
            mean_diff = torch.mean(torch.abs(diff)).item()

            flat_diff = diff.flatten()
            _, top_indices = torch.topk(torch.abs(flat_diff), min(5, flat_diff.numel()))

            m, n = expected_output.shape
            sample_diffs = {}
            for idx in top_indices:
                row = idx.item() // n
                col = idx.item() % n
                sample_diffs[f"({row}, {col})"] = {
                    "expected": expected_output[row, col].item(),
                    "actual": actual_output[row, col].item(),
                    "diff": diff[row, col].item(),
                }

            debug_info = {
                "max_difference": max_diff,
                "mean_difference": mean_diff,
                "sample_differences": sample_diffs,
            }

        return is_close, debug_info

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the number of floating point operations for the problem.

        For log-softmax applied row-wise on an (M, N) matrix:
        Per row:
          - N exp() calls                     : N FLOPs
          - N-1 additions for the sum         : N-1 FLOPs
          - 1 log() of the sum                : 1 FLOP
          - N subtractions (x_i - log(sum))   : N FLOPs
        Total per row: 3N FLOPs (ignoring the -1 and the single log as minor)
        Total: M * 3N

        Args:
            test_case: The test case dictionary

        Returns:
            Number of floating point operations
        """
        M = test_case["rows"]
        N = test_case["cols"]
        return M * 3 * N

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.

        Args:
            test_case: The test case dictionary

        Returns:
            List containing rows M and columns N
        """
        return [test_case["rows"], test_case["cols"]]