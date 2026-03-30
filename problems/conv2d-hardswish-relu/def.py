import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple

from problem import Problem


class conv2d_hardswish_relu(Problem):
    """Exact KernelBench Level 2 Conv2d -> HardSwish -> ReLU port."""

    is_exact = True

    parameters = [
        {"name": "x", "type": "float", "pointer": True, "const": True},
        {"name": "weight", "type": "float", "pointer": True, "const": True},
        {"name": "bias", "type": "float", "pointer": True, "const": True},
        {"name": "output", "type": "float", "pointer": True, "const": False},
        {"name": "batch_size", "type": "size_t", "pointer": False, "const": False},
        {"name": "in_channels", "type": "size_t", "pointer": False, "const": False},
        {"name": "height", "type": "size_t", "pointer": False, "const": False},
        {"name": "width", "type": "size_t", "pointer": False, "const": False},
        {"name": "out_channels", "type": "size_t", "pointer": False, "const": False},
        {"name": "kernel_size", "type": "size_t", "pointer": False, "const": False},
    ]

    def __init__(self):
        super().__init__(name="conv2d-hardswish-relu")

    @staticmethod
    def _make_input(
        batch_size: int,
        in_channels: int,
        height: int,
        width: int,
        seed: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        return torch.rand(
            (batch_size, in_channels, height, width),
            generator=generator,
            dtype=torch.float32,
        ).to(device="cuda", dtype=dtype)

    @staticmethod
    def _make_conv_state(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        seed: int,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=True)
        weight = conv.weight.detach().to(device="cuda", dtype=dtype).contiguous()
        bias = conv.bias.detach().to(device="cuda", dtype=dtype).contiguous()
        return weight, bias

    def reference_solution(
        self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            conv_out = F.conv2d(x, weight, bias)
            return F.relu(F.hardswish(conv_out))

    def generate_test_cases(self) -> List[Dict[str, Any]]:
        dtype = self.param_dtype(0)
        test_configs = [
            (4, 8, 32, 32, 16, 3),
            (8, 8, 48, 48, 24, 3),
            (4, 16, 56, 40, 32, 5),
            (2, 32, 64, 64, 48, 3),
        ]

        test_cases = []
        for batch_size, in_channels, height, width, out_channels, kernel_size in test_configs:
            case_name = (
                f"B={batch_size}, Cin={in_channels}, H={height}, "
                f"W={width}, Cout={out_channels}, K={kernel_size}"
            )
            input_seed = Problem.get_seed(f"{self.name}_{case_name}_input")
            init_seed = Problem.get_seed(f"{self.name}_{case_name}_init")
            test_cases.append(
                {
                    "name": case_name,
                    "batch_size": batch_size,
                    "in_channels": in_channels,
                    "height": height,
                    "width": width,
                    "out_channels": out_channels,
                    "kernel_size": kernel_size,
                    "create_inputs": lambda b=batch_size, cin=in_channels, h=height, w=width, cout=out_channels, k=kernel_size, input_seed=input_seed, init_seed=init_seed, dtype=dtype: (
                        self._make_input(b, cin, h, w, input_seed, dtype),
                        *self._make_conv_state(cin, cout, k, init_seed, dtype),
                    ),
                }
            )
        return test_cases

    def generate_sample(self) -> Dict[str, Any]:
        dtype = self.param_dtype(0)
        return {
            "name": "sample",
            "batch_size": 1,
            "in_channels": 2,
            "height": 4,
            "width": 4,
            "out_channels": 2,
            "kernel_size": 3,
            "create_inputs": lambda d=dtype: (
                torch.tensor(
                    [
                        [
                            [[0.5, -1.0, 1.5, 0.0], [1.0, -0.5, 0.5, -1.5], [0.25, 1.25, -0.75, 0.5], [-1.0, 0.0, 1.0, 1.5]],
                            [[-0.5, 0.25, 1.0, -1.0], [1.5, -1.5, 0.0, 0.5], [0.5, 1.0, -0.25, -0.5], [1.0, -0.75, 0.25, 0.0]],
                        ]
                    ],
                    device="cuda",
                    dtype=d,
                ),
                torch.tensor(
                    [
                        [
                            [[0.25, -0.5, 0.75], [1.0, -1.0, 0.25], [-0.5, 0.5, -0.25]],
                            [[-0.25, 0.75, -0.5], [0.5, -0.25, 0.25], [0.75, -0.5, 0.0]],
                        ],
                        [
                            [[-0.75, 0.25, 0.5], [0.5, -0.5, 0.75], [0.0, 0.25, -1.0]],
                            [[0.25, -0.75, 0.5], [-0.5, 1.0, -0.25], [0.5, 0.25, -0.5]],
                        ],
                    ],
                    device="cuda",
                    dtype=d,
                ),
                torch.tensor([-0.25, 0.5], device="cuda", dtype=d),
            ),
        }

    def verify_result(
        self, expected_output: torch.Tensor, actual_output: torch.Tensor
    ) -> Tuple[bool, Dict[str, Any]]:
        if expected_output.shape != actual_output.shape:
            return False, {
                "message": f"Shape mismatch: expected {tuple(expected_output.shape)}, got {tuple(actual_output.shape)}"
            }

        is_close = torch.allclose(actual_output, expected_output, rtol=3e-4, atol=4e-5)
        if is_close:
            return True, {}

        diff = actual_output - expected_output
        flat_diff = diff.flatten()
        _, top_indices = torch.topk(torch.abs(flat_diff), min(5, flat_diff.numel()))

        out_width = expected_output.shape[-1]
        out_height = expected_output.shape[-2]
        out_channels = expected_output.shape[1]
        sample_diffs = {}
        for idx in top_indices.tolist():
            spatial = idx % (out_height * out_width)
            col = spatial % out_width
            row = spatial // out_width
            channel = (idx // (out_height * out_width)) % out_channels
            batch = idx // (out_channels * out_height * out_width)
            sample_diffs[f"(b={batch}, c={channel}, y={row}, x={col})"] = {
                "expected": expected_output[batch, channel, row, col].item(),
                "actual": actual_output[batch, channel, row, col].item(),
                "diff": diff[batch, channel, row, col].item(),
            }

        debug_info = {
            "max_difference": torch.max(torch.abs(diff)).item(),
            "mean_difference": torch.mean(torch.abs(diff)).item(),
            "expected_positive": int((expected_output > 0).sum().item()),
            "actual_positive": int((actual_output > 0).sum().item()),
            "sample_differences": sample_diffs,
        }
        return False, debug_info

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        batch_size = test_case["batch_size"]
        in_channels = test_case["in_channels"]
        height = test_case["height"]
        width = test_case["width"]
        out_channels = test_case["out_channels"]
        kernel_size = test_case["kernel_size"]
        out_height = height - kernel_size + 1
        out_width = width - kernel_size + 1

        conv_flops = (
            2
            * batch_size
            * out_channels
            * out_height
            * out_width
            * in_channels
            * kernel_size
            * kernel_size
        )
        hardswish_flops = 5 * batch_size * out_channels * out_height * out_width
        relu_flops = batch_size * out_channels * out_height * out_width
        return conv_flops + hardswish_flops + relu_flops

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        return [
            test_case["batch_size"],
            test_case["in_channels"],
            test_case["height"],
            test_case["width"],
            test_case["out_channels"],
            test_case["kernel_size"],
        ]
