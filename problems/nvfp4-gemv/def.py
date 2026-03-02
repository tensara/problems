import torch
from typing import List, Dict, Tuple, Any

from problem import Problem

FP4_AMAX = 6.0
FP8_AMAX = 448.0


class nvfp4_gemv(Problem):

    is_exact = False

    parameters = [
        {"name": "q_a", "type": "uint8_t", "pointer": True, "const": True},
        {"name": "scale_a", "type": "float8", "pointer": True, "const": True},
        {"name": "sf_g_a", "type": "float", "pointer": False, "const": True},
        {"name": "q_x", "type": "uint8_t", "pointer": True, "const": True},
        {"name": "scale_x", "type": "float8", "pointer": True, "const": True},
        {"name": "sf_g_x", "type": "float", "pointer": False, "const": True},
        {"name": "y", "type": "float16", "pointer": True, "const": False},
        {"name": "m", "type": "size_t", "pointer": False, "const": False},
        {"name": "k", "type": "size_t", "pointer": False, "const": False},
    ]

    def __init__(self):
        super().__init__(name="nvfp4-gemv")

    def reference_solution(
        self,
        q_a: torch.Tensor,
        scale_a: torch.Tensor,
        sf_g_a: float,
        q_x: torch.Tensor,
        scale_x: torch.Tensor,
        sf_g_x: float,
    ) -> torch.Tensor:
        from flashinfer.fp4_quantization import e2m1_and_ufp8sf_scale_to_float

        with torch.no_grad():
            sf_a_dec = torch.tensor([1.0 / sf_g_a], device=q_a.device, dtype=torch.float32)
            sf_x_dec = torch.tensor([1.0 / sf_g_x], device=q_x.device, dtype=torch.float32)

            a_deq = e2m1_and_ufp8sf_scale_to_float(q_a, scale_a, sf_a_dec).float()
            x_deq = e2m1_and_ufp8sf_scale_to_float(q_x, scale_x, sf_x_dec).float().squeeze(0)

            return torch.matmul(a_deq, x_deq).to(torch.float16)

    def _make_case(self, m: int, k: int, name: str) -> Dict[str, Any]:
        if k % 16 != 0:
            raise ValueError(f"K must be divisible by 16 for NVFP4, got K={k}")

        seed = Problem.get_seed(f"{self.name}_{name}_M={m}_K={k}")

        def create_inputs(m=m, k=k, seed=seed):
            from flashinfer.fp4_quantization import nvfp4_quantize as _nvfp4_quantize

            g = torch.Generator(device="cuda").manual_seed(seed)
            a = torch.rand((m, k), device="cuda", dtype=self.param_dtype("y"), generator=g) * 2.0 - 1.0
            x = torch.rand((1, k), device="cuda", dtype=self.param_dtype("y"), generator=g) * 2.0 - 1.0

            amax_a = a.float().abs().amax()
            sf_g_a = float((FP4_AMAX * FP8_AMAX) / amax_a)
            sf_g_a_t = torch.tensor([sf_g_a], device=a.device, dtype=torch.float32)

            amax_x = x.float().abs().amax()
            sf_g_x = float((FP4_AMAX * FP8_AMAX) / amax_x)
            sf_g_x_t = torch.tensor([sf_g_x], device=x.device, dtype=torch.float32)

            q_a, scale_a = _nvfp4_quantize(a, sf_g_a_t)
            q_x, scale_x = _nvfp4_quantize(x, sf_g_x_t)

            return q_a, scale_a, sf_g_a, q_x, scale_x, sf_g_x

        return {"name": name, "dims": (m, k), "create_inputs": create_inputs}

    def generate_test_cases(self) -> List[Dict[str, Any]]:
        configs = [
            (1024, 1024, "1024 x 1024"),
            (2048, 2048, "2048 x 2048"),
            (4096, 4096, "4096 x 4096"),
            (8192, 4096, "8192 x 4096"),
            (4096, 8192, "4096 x 8192"),
        ]
        return [self._make_case(m, k, name) for m, k, name in configs]

    def generate_sample(self) -> Dict[str, Any]:
        return self._make_case(32, 32, "sample_32x32")

    def verify_result(
        self, expected_output: torch.Tensor, actual_output: torch.Tensor
    ) -> Tuple[bool, Dict[str, Any]]:
        is_close = torch.allclose(actual_output, expected_output, rtol=2e-2, atol=5e-2)

        debug_info: Dict[str, Any] = {}
        if not is_close:
            diff = actual_output - expected_output
            abs_diff = torch.abs(diff)
            debug_info = {
                "max_difference": abs_diff.max().item(),
                "mean_difference": abs_diff.mean().item(),
            }

        return is_close, debug_info

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        m, k = test_case["dims"]
        return 2 * m * k

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        m, k = test_case["dims"]
        return [m, k]
