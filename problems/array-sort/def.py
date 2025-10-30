import torch
import ctypes
from typing import List, Dict, Tuple, Any
import math

from problem import Problem


class array_sort(Problem):
    """General array sorting problem (integer arrays)."""

    def __init__(self):
        super().__init__(name="array-sort")

    def reference_solution(self, input_array: torch.Tensor) -> torch.Tensor:
        """PyTorch implementation of array sorting on integers."""
        with torch.no_grad():
            sorted_array, _ = torch.sort(input_array)
            return sorted_array

    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        sizes = [16384, 32768, 65536, 131072, 262144]

        test_cases = []
        for size in sizes:
            test_cases.append(
                {
                    "name": f"size_{size}",
                    "size": size,
                    "create_inputs": (
                        lambda s=size: (
                            torch.randint(
                                low=1,
                                high=int(1e9),
                                size=(s,),
                                device="cuda",
                                dtype=torch.int32,
                            ),
                        )
                    ),
                }
            )
        return test_cases

    def generate_sample(self, dtype: torch.dtype = torch.int32) -> Dict[str, Any]:
        """
        Small sample (integers).
        """
        size = 16
        return {
            "name": "Sample (size=16)",
            "size": size,
            "create_inputs": (
                lambda s=size: (
                    torch.randint(0, 100, (s,), device="cuda", dtype=torch.int32),
                )
            ),
        }

    def verify_result(
        self,
        expected_output: torch.Tensor,
        actual_output: torch.Tensor,
        dtype: torch.dtype,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify correctness with exact integer equality.
        """
        is_equal = torch.equal(actual_output, expected_output)

        if is_equal:
            return True, {}

        is_sorted = bool(torch.all(actual_output[:-1] <= actual_output[1:]).item())
        diff_mask = actual_output != expected_output
        idx = torch.nonzero(diff_mask).flatten()

        sample_diffs = {}
        for i in idx[:5].tolist():
            sample_diffs[f"index_{i}"] = {
                "expected": int(expected_output[i].item()),
                "actual": int(actual_output[i].item()),
            }

        debug_info = {
            "is_sorted": is_sorted,
            "total_mismatches": int(idx.numel()),
            "sample_differences": sample_diffs,
        }
        return False, debug_info

    def get_function_signature(self) -> Dict[str, Any]:
        """
        C signature:
            void sort_kernel(const int32_t* input, int32_t* output, size_t size);
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_int32),  # input_array
                ctypes.POINTER(ctypes.c_int32),  # output_array
                ctypes.c_size_t,  # size
            ],
            "restype": None,
        }

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """Extra parameters for CUDA call (just the size)."""
        return [test_case["size"]]
