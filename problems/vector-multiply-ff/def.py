import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem

MODULUS = 0x78000001  # BabyBear modulus

class vector_multiply_babybear(Problem):
    """Vector multiplication over the BabyBear field."""

    def __init__(self):
        super().__init__(
            name="vector-multiply-babybear"
        )

    def reference_solution(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        PyTorch reference for BabyBear vector multiplication.
        Performs element-wise multiplication mod BabyBear prime.
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.int32):
            prod = A.long() * B.long()
            return (prod % MODULUS).int()

    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        sizes = [
            ("n = 2^20", 1048576),
            ("n = 2^22", 4194304),
            ("n = 2^24", 16777216),
            ("n = 2^25", 33554432),
        ]

        return [
            {
                "name": name,
                "dims": (size,),
                "create_inputs": lambda size=size: (
                    torch.randint(0, MODULUS, (size,), device="cuda", dtype=torch.int64).to(torch.uint32),
                    torch.randint(0, MODULUS, (size,), device="cuda", dtype=torch.int64).to(torch.uint32),
                )

            }
            for name, size in sizes
        ]

    def generate_sample(self, dtype: torch.dtype = torch.uint32) -> Dict[str, Any]:
        name = "Sample (n = 8)"
        size = 8
        return {
            "name": name,
            "dims": (size,),
            "create_inputs": lambda size=size: (
                torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], device="cuda", dtype=torch.int64).to(torch.uint32),
                torch.tensor([8, 7, 6, 5, 4, 3, 2, 1], device="cuda", dtype=torch.int64).to(torch.uint32),
            )
        }
   

    def verify_result(self, expected_output: torch.Tensor,
                      actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        is_equal = torch.equal(expected_output, actual_output)
        debug_info = {}

        if not is_equal:
            diff = (actual_output - expected_output).abs()
            max_diff = diff.max().item()
            debug_info = {
                "max_difference": max_diff,
                "num_mismatches": (diff > 0).sum().item(),
            }

        return is_equal, debug_info

    def get_function_signature(self) -> Dict[str, Any]:
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_uint32),  # input A
                ctypes.POINTER(ctypes.c_uint32),  # input B
                ctypes.POINTER(ctypes.c_uint32),  # output
                ctypes.c_size_t                   # N
            ],
            "restype": None
        }

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        # One multiply and one mod per element
        return 2 * test_case["dims"][0]

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        return [test_case["dims"][0]]
