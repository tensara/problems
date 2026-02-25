import torch
from typing import List, Dict, Tuple, Any, Optional

from problem import Problem

FP4_AMAX = 6.0
FP8_AMAX = 448.0


class nvfp4_gemm(Problem):

    is_exact = False

    parameters = [
        {"name": "q_a", "type": "uint8_t", "pointer": True, "const": True},
        {"name": "scale_a", "type": "float8", "pointer": True, "const": True},
        {"name": "sf_g_a", "type": "float", "pointer": False, "const": True},
        {"name": "q_b", "type": "uint8_t", "pointer": True, "const": True},
        {"name": "scale_b", "type": "float8", "pointer": True, "const": True},
        {"name": "sf_g_b", "type": "float", "pointer": False, "const": True},
        {"name": "c", "type": "float16", "pointer": True, "const": False},
        {"name": "m", "type": "size_t", "pointer": False, "const": False},
        {"name": "n", "type": "size_t", "pointer": False, "const": False},
        {"name": "k", "type": "size_t", "pointer": False, "const": False},
    ]

    def __init__(self):
        super().__init__(name="nvfp4-gemm")

    def reference_solution(
        self,
        q_a: torch.Tensor,
        scale_a: torch.Tensor,
        sf_g_a: float,
        q_b: torch.Tensor,
        scale_b: torch.Tensor,
        sf_g_b: float,
    ) -> torch.Tensor:
        from torch.nn.functional import scaled_mm, ScalingType, SwizzleType  

        with torch.no_grad():
            sf_a_dec = torch.tensor([1.0 / sf_g_a], device=q_a.device, dtype=torch.float32)
            sf_b_dec = torch.tensor([1.0 / sf_g_b], device=q_b.device, dtype=torch.float32)

            scale_a_flat = scale_a.contiguous().view(torch.float8_e4m3fn).flatten()
            scale_b_flat = scale_b.contiguous().view(torch.float8_e4m3fn).flatten()

            c_scaled = scaled_mm(
                q_a.view(torch.float4_e2m1fn_x2),  
                q_b.view(torch.float4_e2m1fn_x2).t(),
                scale_a=[scale_a_flat, sf_a_dec],  
                scale_recipe_a=[ScalingType.BlockWise1x16, ScalingType.TensorWise],  
                scale_b=[scale_b_flat, sf_b_dec],
                scale_recipe_b=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
                swizzle_a=[SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],  
                swizzle_b=[SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],  
                output_dtype=torch.float16,  
            )

            return c_scaled

    def _make_case(self, m: int, n: int, k: int, name: str) -> Dict[str, Any]:
        if k % 16 != 0:
            raise ValueError(f"K must be divisible by 16 for NVFP4, got K={k}")

        seed = Problem.get_seed(f"{self.name}_{name}_M={m}_N={n}_K={k}")

        def create_inputs(m=m, n=n, k=k, seed=seed):
            from flashinfer.fp4_quantization import nvfp4_quantize as _nvfp4_quantize

            g = torch.Generator(device="cuda").manual_seed(seed)
            a = torch.rand((m, k), device="cuda", dtype=self.param_dtype("c"), generator=g) * 2.0 - 1.0
            b = torch.rand((n, k), device="cuda", dtype=self.param_dtype("c"), generator=g) * 2.0 - 1.0

            amax_a = a.float().abs().amax()
            sf_g_a = float((FP4_AMAX * FP8_AMAX) / amax_a)
            sf_g_a_t = torch.tensor([sf_g_a], device=a.device, dtype=torch.float32)

            amax_b = b.float().abs().amax()
            sf_g_b = float((FP4_AMAX * FP8_AMAX) / amax_b)
            sf_g_b_t = torch.tensor([sf_g_b], device=b.device, dtype=torch.float32)

            q_a, scale_a = _nvfp4_quantize(a, sf_g_a_t)
            q_b, scale_b = _nvfp4_quantize(b, sf_g_b_t)
            return q_a, scale_a, sf_g_a, q_b, scale_b, sf_g_b

        return {"name": name, "dims": (m, n, k), "create_inputs": create_inputs}

    def generate_test_cases(self) -> List[Dict[str, Any]]:
        configs = [
            (1024, 1024, 1024, "1024 x 1024 x 1024"),
            (2048, 1024, 2048, "2048 x 1024 x 2048"),
            (4096, 2048, 4096, "4096 x 2048 x 4096"),
            (4096, 4096, 4096, "4096 x 4096 x 4096"),
            (8192, 4096, 8192, "8192 x 4096 x 8192"),
        ]
        return [self._make_case(m, n, k, name) for m, n, k, name in configs]

    def generate_sample(self) -> Dict[str, Any]:
        return self._make_case(32, 32, 32, "sample_32x32x32")

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
        m, n, k = test_case["dims"]
        return 2 * m * n * k

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        m, n, k = test_case["dims"]
        return [m, n, k]
