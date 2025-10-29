import torch
import ctypes
from typing import List, Dict, Tuple, Any
import math

from problem import Problem


def _next_power_of_two(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


class array_sort(Problem):
    """General array sorting problem."""

    def __init__(self):
        super().__init__(name="array-sort")

    def reference_solution(self, input_array: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of array sorting.
        """
        with torch.no_grad():
            sorted_array, _ = torch.sort(input_array)
            return sorted_array

    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases whose sizes are powers of two.
        We take your original sizes and round each up to the next power of two.
        """
        base_sizes = [1000, 5000, 10000, 25000, 50000]
        pow2_sizes = sorted({_next_power_of_two(s) for s in base_sizes})
        # Example: {1024, 8192, 16384, 32768, 65536}

        test_cases = []
        for size in pow2_sizes:
            test_cases.append({
                "name": f"size_{size}",
                "size": size,
                "create_inputs": (lambda s=size, dt=dtype: (
                    torch.rand(s, device="cuda", dtype=dt) * 1000.0,
                ))
            })

        return test_cases

    def generate_sample(self, dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
        """
        Generate a single sample test case (power of two).
        """
        size = 16
        return {
            "name": "Sample (size=16)",
            "size": size,
            "create_inputs": (lambda s=size, dt=dtype: (
                torch.rand(s, device="cuda", dtype=dt) * 1000.0,
            ))
        }

    def verify_result(self, expected_output: torch.Tensor,
                      actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the sorting result is correct.
        """
        is_close = torch.allclose(actual_output, expected_output, rtol=1e-5, atol=1e-5)

        debug_info = {}
        if not is_close:
            # Check if array is sorted
            is_sorted = torch.all(actual_output[:-1] <= actual_output[1:])

            # Find first mismatch
            diff = actual_output - expected_output
            nonzero_diff = torch.nonzero(torch.abs(diff) > 1e-5)

            sample_diffs = {}
            if nonzero_diff.size(0) > 0:
                for i in range(min(5, nonzero_diff.size(0))):
                    idx = nonzero_diff[i, 0].item()
                    sample_diffs[f"index_{idx}"] = {
                        "expected": expected_output[idx].item(),
                        "actual": actual_output[idx].item(),
                        "diff": diff[idx].item()
                    }

            debug_info = {
                "is_sorted": is_sorted.item(),
                "sample_differences": sample_diffs,
                "total_mismatches": nonzero_diff.size(0)
            }

        return is_close, debug_info

    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the array sorting solution.
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # input_array
                ctypes.POINTER(ctypes.c_float),  # output_array
                ctypes.c_size_t,                 # size
            ],
            "restype": None
        }

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the number of floating point operations for the problem.

        IMPORTANT: Comments are required. Outline the FLOPs calculation.

        We estimate O(n log n) comparisons; assume ~3 FLOPs per comparison
        (1 compare + ~2 ops amortized for swaps/moves).
        """
        size = test_case["size"]
        return int(size * math.log2(size) * 3)

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Extra parameters to pass to the CUDA solution.
        """
        size = test_case["size"]
        return [size]
