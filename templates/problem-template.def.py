import torch
from typing import Any, Dict, List, Tuple

from problem import Problem


class template_problem(Problem):
    is_exact = False

    parameters = [
        {"name": "input", "type": "float", "pointer": True, "const": True},
        {"name": "output", "type": "float", "pointer": True, "const": False},
        {"name": "n", "type": "size_t", "pointer": False, "const": False},
    ]

    def __init__(self):
        super().__init__(name="template-problem")

    def reference_solution(self, input_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=input_tensor.dtype):
            return input_tensor

    def generate_test_cases(self) -> List[Dict[str, Any]]:
        dtype = self.param_dtype("input")
        test_cases = []
        for size in (256, 1024, 4096):
            seed = Problem.get_seed(f"{self.name}_{size}")
            test_cases.append(
                {
                    "name": f"n={size}",
                    "n": size,
                    "seed": seed,
                    "create_inputs": lambda size=size, seed=seed, dtype=dtype: (
                        torch.randn(
                            (size,),
                            device="cuda",
                            dtype=dtype,
                            generator=torch.Generator(device="cuda").manual_seed(seed),
                        ),
                    ),
                }
            )
        return test_cases

    def generate_sample(self) -> Dict[str, Any]:
        dtype = self.param_dtype("input")
        return {
            "name": "n=8",
            "n": 8,
            "create_inputs": lambda dtype=dtype: (
                torch.tensor([1, -2, 3, -4, 5, -6, 7, -8], device="cuda", dtype=dtype),
            ),
        }

    def verify_result(
        self, expected_output: torch.Tensor, actual_output: torch.Tensor
    ) -> Tuple[bool, Dict[str, Any]]:
        is_close = torch.allclose(actual_output, expected_output, rtol=1e-5, atol=1e-6)
        if is_close:
            return True, {}

        diff = actual_output - expected_output
        return False, {
            "max_difference": torch.max(torch.abs(diff)).item(),
            "mean_difference": torch.mean(torch.abs(diff)).item(),
        }

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        return test_case["n"]

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        return [test_case["n"]]
