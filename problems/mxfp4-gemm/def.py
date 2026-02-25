import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any

from problem import Problem

BLOCK_SIZE = 32


class mxfp4_gemm(Problem):

    is_exact = False

    parameters = [
        {"name": "q_a", "type": "uint8_t", "pointer": True, "const": True},
        {"name": "scale_a", "type": "uint8_t", "pointer": True, "const": True},
        {"name": "q_b", "type": "uint8_t", "pointer": True, "const": True},
        {"name": "scale_b", "type": "uint8_t", "pointer": True, "const": True},
        {"name": "c", "type": "float", "pointer": True, "const": False},
        {"name": "m", "type": "size_t", "pointer": False, "const": False},
        {"name": "n", "type": "size_t", "pointer": False, "const": False},
        {"name": "k", "type": "size_t", "pointer": False, "const": False},
    ]

    def __init__(self):
        super().__init__(name="mxfp4-gemm")

    @staticmethod
    def _mx_tensor_api():
        try:
            from torchao.prototype.mx_formats.mx_tensor import to_mx
        except Exception as e:
            raise RuntimeError(
                "TorchAO MXTensor APIs are required. Install a torchao build with "
                "torchao.prototype.mx_formats.mx_tensor support."
            ) from e
        return to_mx

    def reference_solution(self, q_a: torch.Tensor, scale_a: torch.Tensor, q_b: torch.Tensor, scale_b: torch.Tensor) -> torch.Tensor:
        if not hasattr(F, "scaled_mm"):
            raise RuntimeError("torch.nn.functional.scaled_mm is required for this problem.")

        with torch.no_grad():
            a_q = q_a.view(torch.float4_e2m1fn_x2)
            b_q = q_b.view(torch.float4_e2m1fn_x2)
            s_a = scale_a.view(torch.float8_e8m0fnu).flatten()
            s_b = scale_b.view(torch.float8_e8m0fnu).flatten()

            return F.scaled_mm(
                a_q,
                b_q.t(),
                scale_a=s_a,
                scale_recipe_a=F.ScalingType.BlockWise1x32,
                swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
                scale_b=s_b,
                scale_recipe_b=F.ScalingType.BlockWise1x32,
                swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
                output_dtype=torch.float32,
            )

    def _make_case(self, m: int, n: int, k: int, name: str) -> Dict[str, Any]:
        if k % BLOCK_SIZE != 0:
            raise ValueError(f"K must be divisible by {BLOCK_SIZE}, got K={k}")
        if n % BLOCK_SIZE != 0:
            raise ValueError(f"N must be divisible by {BLOCK_SIZE}, got N={n}")

        seed = Problem.get_seed(f"{self.name}_{name}_M={m}_N={n}_K={k}")

        def create_inputs(m=m, n=n, k=k, seed=seed):
            g = torch.Generator(device="cuda").manual_seed(seed)

            a = torch.randn((m, k), device="cuda", dtype=self.param_dtype("c"), generator=g)
            b = torch.randn((n, k), device="cuda", dtype=self.param_dtype("c"), generator=g)

            to_mx = self._mx_tensor_api()
            scale_a_e8m0, a_lp = to_mx(a, torch.float4_e2m1fn_x2, BLOCK_SIZE, is_swizzled_scales=True)
            scale_b_e8m0, b_lp = to_mx(b, torch.float4_e2m1fn_x2, BLOCK_SIZE, is_swizzled_scales=True)

            q_a = a_lp.contiguous().view(torch.uint8)
            scale_a = scale_a_e8m0.contiguous().view(torch.uint8)
            q_b = b_lp.contiguous().view(torch.uint8)
            scale_b = scale_b_e8m0.contiguous().view(torch.uint8)

            return q_a, scale_a, q_b, scale_b

        return {"name": name, "dims": (m, n, k), "create_inputs": create_inputs}

    def generate_test_cases(self) -> List[Dict[str, Any]]:
        configs = [
            (1024, 1024, 1024, "1024 x 1024 x 1024"),
            (2048, 1024, 2048, "2048 x 1024 x 2048"),
            (4096, 2048, 4096, "4096 x 2048 x 4096"),
            (4096, 4096, 4096, "4096 x 4096 x 4096"),
            (8192, 4096, 8192, "8192 x 4096 x 8192"),
        ]
        return [self._make_case(m, n, k, name) for m, k, n, name in configs]

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
