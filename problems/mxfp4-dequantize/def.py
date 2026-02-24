import torch
from typing import List, Dict, Tuple, Any

from problem import Problem

BLOCK_SIZE = 32


class mxfp4_dequantize(Problem):

    is_exact = False

    parameters = [
        {"name": "q", "type": "uint8_t", "pointer": True, "const": True},
        {"name": "scale", "type": "uint8_t", "pointer": True, "const": True},
        {"name": "out", "type": "float", "pointer": True, "const": False},
        {"name": "m", "type": "size_t", "pointer": False, "const": False},
        {"name": "k", "type": "size_t", "pointer": False, "const": False},
    ]

    def __init__(self):
        super().__init__(name="mxfp4-dequantize")

    @staticmethod
    def _mx_tensor_api():
        try:
            from torchao.prototype.mx_formats.mx_tensor import to_mx, to_dtype
        except Exception as e:
            raise RuntimeError(
                "TorchAO MXTensor APIs are required. Install a torchao build with "
                "torchao.prototype.mx_formats.mx_tensor support."
            ) from e
        return to_mx, to_dtype

    def reference_solution(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        _, to_dtype = self._mx_tensor_api()

        with torch.no_grad():
            scale_e8m0 = scale.view(torch.float8_e8m0fnu)
            return to_dtype(q, scale_e8m0, torch.float4_e2m1fn_x2, BLOCK_SIZE, torch.float32).float()

    def _make_case(self, m: int, k: int, name: str) -> Dict[str, Any]:
        if k % BLOCK_SIZE != 0:
            raise ValueError(f"K must be divisible by {BLOCK_SIZE}, got K={k}")

        seed = Problem.get_seed(f"{self.name}_{name}_M={m}_K={k}")

        def create_inputs(m=m, k=k, seed=seed):
            to_mx, _ = self._mx_tensor_api()

            g = torch.Generator(device="cuda").manual_seed(seed)
            a = torch.rand((m, k), device="cuda", dtype=torch.float32, generator=g) * 2.0 - 1.0
            scale_e8m0, data_lp = to_mx(a, torch.float4_e2m1fn_x2, BLOCK_SIZE)
            q = data_lp.contiguous().view(torch.uint8)
            scale = scale_e8m0.contiguous().view(torch.uint8)
            return q, scale

        return {"name": name, "dims": (m, k), "create_inputs": create_inputs}

    def generate_test_cases(self) -> List[Dict[str, Any]]:
        configs = [
            (1024, 1024, "1024 x 1024"),
            (2048, 2048, "2048 x 2048"),
            (4096, 4096, "4096 x 4096"),
            (8192, 4096, "8192 x 4096"),
            (4096, 8192, "4096 x 8192")
        ]
        return [self._make_case(m, k, name) for m, k, name in configs]

    def generate_sample(self) -> Dict[str, Any]:
        return self._make_case(32, 32, "sample_32x32")

    def verify_result(
        self, expected_output: torch.Tensor, actual_output: torch.Tensor
    ) -> Tuple[bool, Dict[str, Any]]:
        is_close = torch.allclose(actual_output, expected_output, rtol=1e-3, atol=1e-3)

        debug_info: Dict[str, Any] = {}
        if not is_close:
            diff = actual_output - expected_output
            abs_diff = torch.abs(diff)
            debug_info = {
                "max_difference": abs_diff.max().item(),
                "mean_difference": abs_diff.mean().item(),
            }

        return is_close, debug_info

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        m, k = test_case["dims"]
        return [m, k]
