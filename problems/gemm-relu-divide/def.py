import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple

from problem import Problem


class gemm_relu_divide(Problem):
    """Exact KernelBench Level 2 GEMM -> ReLU -> divide port."""

    is_exact = True

    parameters = [
        {"name": "x", "type": "float", "pointer": True, "const": True},
        {"name": "weight", "type": "float", "pointer": True, "const": True},
        {"name": "bias", "type": "float", "pointer": True, "const": True},
        {"name": "divisor", "type": "float", "pointer": False, "const": True},
        {"name": "output", "type": "float", "pointer": True, "const": False},
        {"name": "batch_size", "type": "size_t", "pointer": False, "const": False},
        {"name": "in_features", "type": "size_t", "pointer": False, "const": False},
        {"name": "out_features", "type": "size_t", "pointer": False, "const": False},
    ]

    def __init__(self):
        super().__init__(name="gemm-relu-divide")

    @staticmethod
    def _make_input(
        batch_size: int, in_features: int, seed: int, dtype: torch.dtype
    ) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        return torch.rand((batch_size, in_features), generator=generator, dtype=torch.float32).to(
            device="cuda", dtype=dtype
        )

    @staticmethod
    def _make_linear_state(
        in_features: int, out_features: int, seed: int, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            linear = torch.nn.Linear(in_features, out_features, bias=True)
        weight = linear.weight.detach().to(device="cuda", dtype=dtype).contiguous()
        bias = linear.bias.detach().to(device="cuda", dtype=dtype).contiguous()
        return weight, bias

    def reference_solution(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        divisor: float,
    ) -> torch.Tensor:
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            logits = F.linear(x, weight, bias)
            return F.relu(logits) / divisor

    def generate_test_cases(self) -> List[Dict[str, Any]]:
        dtype = self.param_dtype(0)
        divisor = 2.0
        test_configs = [
            (64, 512, 384),
            (128, 1024, 768),
            (192, 1536, 1024),
            (256, 2048, 1536),
        ]

        test_cases = []
        for batch_size, in_features, out_features in test_configs:
            case_name = f"B={batch_size}, I={in_features}, O={out_features}"
            input_seed = Problem.get_seed(f"{self.name}_{case_name}_input")
            init_seed = Problem.get_seed(f"{self.name}_{case_name}_init")
            test_cases.append(
                {
                    "name": case_name,
                    "batch_size": batch_size,
                    "in_features": in_features,
                    "out_features": out_features,
                    "divisor": divisor,
                    "create_inputs": lambda b=batch_size, i=in_features, o=out_features, d=divisor, input_seed=input_seed, init_seed=init_seed, dtype=dtype: (
                        self._make_input(b, i, input_seed, dtype),
                        *self._make_linear_state(i, o, init_seed, dtype),
                        d,
                    ),
                }
            )
        return test_cases

    def generate_sample(self) -> Dict[str, Any]:
        dtype = self.param_dtype(0)
        return {
            "name": "sample",
            "batch_size": 2,
            "in_features": 4,
            "out_features": 3,
            "divisor": 2.0,
            "create_inputs": lambda d=dtype: (
                torch.tensor(
                    [[1.0, 0.5, -1.0, 2.0], [-0.5, 1.5, 0.25, -2.0]],
                    device="cuda",
                    dtype=d,
                ),
                torch.tensor(
                    [
                        [0.5, -1.0, 0.75, 1.5],
                        [-0.25, 0.5, 1.0, -0.75],
                        [1.25, -0.5, -1.0, 0.25],
                    ],
                    device="cuda",
                    dtype=d,
                ),
                torch.tensor([0.5, -0.75, 0.25], device="cuda", dtype=d),
                2.0,
            ),
        }

    def verify_result(
        self, expected_output: torch.Tensor, actual_output: torch.Tensor
    ) -> Tuple[bool, Dict[str, Any]]:
        if expected_output.shape != actual_output.shape:
            return False, {
                "message": f"Shape mismatch: expected {tuple(expected_output.shape)}, got {tuple(actual_output.shape)}"
            }

        is_close = torch.allclose(actual_output, expected_output, rtol=2e-4, atol=2e-5)
        if is_close:
            return True, {}

        diff = actual_output - expected_output
        flat_diff = diff.flatten()
        _, top_indices = torch.topk(torch.abs(flat_diff), min(5, flat_diff.numel()))

        rows, cols = expected_output.shape
        sample_diffs = {}
        for idx in top_indices.tolist():
            row = idx // cols
            col = idx % cols
            sample_diffs[f"({row}, {col})"] = {
                "expected": expected_output[row, col].item(),
                "actual": actual_output[row, col].item(),
                "diff": diff[row, col].item(),
            }

        debug_info = {
            "max_difference": torch.max(torch.abs(diff)).item(),
            "mean_difference": torch.mean(torch.abs(diff)).item(),
            "expected_nonzero": int((expected_output > 0).sum().item()),
            "actual_nonzero": int((actual_output > 0).sum().item()),
            "sample_differences": sample_diffs,
        }
        return False, debug_info

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        batch_size = test_case["batch_size"]
        in_features = test_case["in_features"]
        out_features = test_case["out_features"]

        # Matrix multiply: 2 * B * I * O
        # Bias add: B * O
        # ReLU: B * O
        # Divide: B * O
        return (2 * batch_size * in_features * out_features) + (3 * batch_size * out_features)

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        return [
            test_case["batch_size"],
            test_case["in_features"],
            test_case["out_features"],
        ]
