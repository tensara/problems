import torch
from typing import List, Dict, Tuple, Any

from problem import Problem

BLOCK_SIZE = 32


class mxfp8_quantize(Problem):

    is_exact = False

    parameters = [
        {"name": "a", "type": "float", "pointer": True, "const": True},
        {"name": "q", "type": "uint8_t", "pointer": True, "const": False},
        {"name": "scale", "type": "uint8_t", "pointer": True, "const": False},
        {"name": "m", "type": "size_t", "pointer": False, "const": False},
        {"name": "k", "type": "size_t", "pointer": False, "const": False},
    ]

    def __init__(self):
        super().__init__(name="mxfp8-quantize")

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

    def reference_solution(self, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        to_mx, _ = self._mx_tensor_api()

        with torch.no_grad():
            scale_e8m0, data_lp = to_mx(a, torch.float8_e4m3fn, BLOCK_SIZE)
            q_bytes = data_lp.contiguous().view(torch.uint8)
            scale_bytes = scale_e8m0.contiguous().view(torch.uint8)
            return q_bytes, scale_bytes

    def _make_case(self, m: int, k: int, name: str) -> Dict[str, Any]:
        if k % BLOCK_SIZE != 0:
            raise ValueError(f"K must be divisible by {BLOCK_SIZE}, got K={k}")

        seed = Problem.get_seed(f"{self.name}_{name}_M={m}_K={k}")

        def create_inputs(m=m, k=k, seed=seed):
            g = torch.Generator(device="cuda").manual_seed(seed)
            a = torch.rand((m, k), device="cuda", dtype=self.param_dtype(0), generator=g) * 2.0 - 1.0
            return (a,)

        return {
            "name": name,
            "dims": (m, k),
            "create_inputs": create_inputs,
        }

    def generate_test_cases(self) -> List[Dict[str, Any]]:
        configs = [
            (1024, 1024, "1024 x 1024"),
            (2048, 2048, "2048 x 2048"),
            (4096, 8192, "4096 x 8192"),
            (8192, 4096, "8192 x 4096"),
        ]
        return [self._make_case(m, k, name) for m, k, name in configs]

    def generate_sample(self) -> Dict[str, Any]:
        return self._make_case(32, 32, "sample_32x32")

    def _dequantize(self, q_bytes: torch.Tensor, scale_bytes: torch.Tensor) -> torch.Tensor:
        _, to_dtype = self._mx_tensor_api()

        scale_e8m0 = scale_bytes.view(torch.float8_e8m0fnu)
        q_lp = q_bytes.view(torch.float8_e4m3fn)
        return to_dtype(q_lp, scale_e8m0, torch.float8_e4m3fn, BLOCK_SIZE, torch.float32).float()

    def verify_result(
        self,
        expected_output: Tuple[torch.Tensor, torch.Tensor],
        actual_output: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[bool, Dict[str, Any]]:
        q_ref, sf_ref = expected_output
        q_actual, sf_actual = actual_output

        dequant_ref = self._dequantize(q_ref, sf_ref)
        dequant_actual = self._dequantize(q_actual, sf_actual)

        is_close = torch.allclose(dequant_ref, dequant_actual, rtol=1e-3, atol=1e-3)

        debug_info: Dict[str, Any] = {}
        if not is_close:
            diff = dequant_actual - dequant_ref
            abs_diff = torch.abs(diff)
            debug_info = {
                "max_difference": abs_diff.max().item(),
                "mean_difference": abs_diff.mean().item(),
            }

        return is_close, debug_info

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        m, k = test_case["dims"]
        return [m, k]
