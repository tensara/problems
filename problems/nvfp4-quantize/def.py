import torch
from typing import List, Dict, Tuple, Any

from problem import Problem

FP4_AMAX = 6.0
FP8_AMAX = 448.0

class nvfp4_quantize(Problem):

    is_exact = False

    parameters = [
        {"name": "a", "type": "float16", "pointer": True, "const": True},
        {"name": "sf_g", "type": "float", "pointer": False, "const": True},
        {"name": "q", "type": "float8", "pointer": True, "const": False},
        {"name": "scale", "type": "float4", "pointer": True, "const": False},
        {"name": "m", "type": "size_t", "pointer": False, "const": False},
        {"name": "k", "type": "size_t", "pointer": False, "const": False},
    ]

    def __init__(self):
        super().__init__(name="nvfp4-quantize")

    def reference_solution(self, a: torch.Tensor, sf_g: float) -> Tuple[torch.Tensor, torch.Tensor]:
        from flashinfer.fp4_quantization import nvfp4_quantize as _nvfp4_quantize

        self._last_global_sf = sf_g

        sf_g_t = torch.tensor([sf_g], device=a.device, dtype=self.param_dtype("sf_g"))

        with torch.no_grad():
            q, sf = _nvfp4_quantize(a, sf_g_t)
            return q, sf

    def _make_case(self, m: int, k: int, name: str) -> Dict[str, Any]:
        seed = Problem.get_seed(f"{self.name}_{name}_M={m}_K={k}")

        def create_inputs(m=m, k=k, seed=seed):
            g = torch.Generator(device="cuda").manual_seed(seed)
            a = torch.rand((m, k), device="cuda", dtype=self.param_dtype(0), generator=g) * 2.0 - 1.0
            amax = a.float().abs().amax()
            sf_g = float((FP4_AMAX * FP8_AMAX) / amax)
            return a, sf_g

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
        return self._make_case(16, 16, "sample_16x16")

    def verify_result(
        self, expected_output: Tuple[torch.Tensor, torch.Tensor],
        actual_output: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[bool, Dict[str, Any]]:
        from flashinfer.fp4_quantization import e2m1_and_ufp8sf_scale_to_float

        q_ref, sf_ref = expected_output
        q_actual, sf_actual = actual_output

        s_dec = torch.tensor(
            [1.0 / self._last_global_sf], device=q_ref.device, dtype=self.param_dtype("sf_g")
        )

        dequant_ref = e2m1_and_ufp8sf_scale_to_float(q_ref, sf_ref, s_dec).float()
        dequant_actual = e2m1_and_ufp8sf_scale_to_float(q_actual, sf_actual, s_dec).float()

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
