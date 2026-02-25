import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any, Optional

from problem import Problem

FP4_AMAX = 6.0
FP8_AMAX = 448.0


class nvfp4_gemv(Problem):

    is_exact = False

    parameters = [
        {"name": "q_a", "type": "uint8_t", "pointer": True, "const": True},
        {"name": "scale_a", "type": "float8", "pointer": True, "const": True},
        {"name": "q_x", "type": "uint8_t", "pointer": True, "const": True},
        {"name": "scale_x", "type": "float8", "pointer": True, "const": True},
        {"name": "y", "type": "float", "pointer": True, "const": False},
        {"name": "m", "type": "size_t", "pointer": False, "const": False},
        {"name": "k", "type": "size_t", "pointer": False, "const": False},
        {"name": "sf_g_a", "type": "float", "pointer": False, "const": True},
        {"name": "sf_g_x", "type": "float", "pointer": False, "const": True},
    ]

    def __init__(self):
        super().__init__(name="nvfp4-gemv")

    @staticmethod
    def _first_enum_member(enum_cls: Any, candidates: List[str]) -> Any:
        for name in candidates:
            if hasattr(enum_cls, name):
                return getattr(enum_cls, name)
        return None

    @staticmethod
    def _dequantize(q: torch.Tensor, scale: torch.Tensor, sf_g: float) -> torch.Tensor:
        from flashinfer.fp4_quantization import e2m1_and_ufp8sf_scale_to_float

        s_dec = torch.tensor([1.0 / sf_g], device=q.device, dtype=torch.float32)
        return e2m1_and_ufp8sf_scale_to_float(q, scale, s_dec).float()

    @staticmethod
    def _quantize_fp8_tensorwise(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        amax = torch.clamp(x.abs().amax(), min=1e-8)
        scale = (amax / FP8_AMAX).float()
        q = (x / scale).to(torch.float8_e4m3fn)
        return q, scale.view(1)

    @staticmethod
    def _call_scaled_mm(
        a_q: torch.Tensor,
        b_q: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        recipe_a: Any,
        recipe_b: Any,
    ) -> Optional[torch.Tensor]:
        if not hasattr(F, "scaled_mm"):
            return None

        kwargs: Dict[str, Any] = {"scale_a": scale_a, "scale_b": scale_b}
        if recipe_a is not None:
            kwargs["scale_recipe_a"] = recipe_a
        if recipe_b is not None:
            kwargs["scale_recipe_b"] = recipe_b

        variants = [kwargs, {**kwargs, "scale_a": [scale_a], "scale_b": [scale_b]}]
        for variant in variants:
            try:
                return F.scaled_mm(a_q, b_q, out_dtype=torch.float32, **variant)
            except TypeError:
                try:
                    return F.scaled_mm(a_q, b_q, output_dtype=torch.float32, **variant)
                except Exception:
                    pass
            except Exception:
                pass
        return None

    def reference_solution(
        self,
        q_a: torch.Tensor,
        scale_a: torch.Tensor,
        q_x: torch.Tensor,
        scale_x: torch.Tensor,
        sf_g_a: float,
        sf_g_x: float,
    ) -> torch.Tensor:
        if not torch.isfinite(torch.tensor(sf_g_a, device=q_a.device)) or sf_g_a == 0.0:
            raise ValueError(f"sf_g_a must be finite and non-zero, got sf_g_a={sf_g_a}")
        if not torch.isfinite(torch.tensor(sf_g_x, device=q_x.device)) or sf_g_x == 0.0:
            raise ValueError(f"sf_g_x must be finite and non-zero, got sf_g_x={sf_g_x}")

        with torch.no_grad():
            a_dequant = self._dequantize(q_a, scale_a, sf_g_a)
            x_dequant = self._dequantize(q_x, scale_x, sf_g_x)

            if a_dequant.ndim != 2 or x_dequant.ndim != 2:
                raise ValueError(
                    f"Expected 2D dequantized inputs, got A={tuple(a_dequant.shape)}, X={tuple(x_dequant.shape)}"
                )

            m, k = a_dequant.shape
            if k % 16 != 0:
                raise ValueError(f"K must be divisible by 16 for NVFP4, got K={k}")
            if x_dequant.shape != (1, k):
                raise ValueError(f"Expected X shape (1,{k}), got {tuple(x_dequant.shape)}")

            a_q, s_a = self._quantize_fp8_tensorwise(a_dequant)
            x_q, s_x = self._quantize_fp8_tensorwise(x_dequant.view(k, 1))

            scaling_type = getattr(F, "ScalingType", None)
            recipe = None
            if scaling_type is not None:
                recipe = self._first_enum_member(
                    scaling_type, ["TensorWise", "TENSORWISE", "PER_TENSOR"]
                )

            out = self._call_scaled_mm(a_q, x_q, s_a, s_x, recipe, recipe)
            if out is None:
                out = torch.matmul(a_dequant, x_dequant.view(k)).view(m, 1)

            return out.view(m)

    def _make_case(self, m: int, k: int, name: str) -> Dict[str, Any]:
        if k % 16 != 0:
            raise ValueError(f"K must be divisible by 16 for NVFP4, got K={k}")

        seed = Problem.get_seed(f"{self.name}_{name}_M={m}_K={k}")

        def create_inputs(m=m, k=k, seed=seed):
            from flashinfer.fp4_quantization import nvfp4_quantize as _nvfp4_quantize

            g = torch.Generator(device="cuda").manual_seed(seed)
            a = torch.rand((m, k), device="cuda", dtype=torch.float16, generator=g) * 2.0 - 1.0
            x = torch.rand((1, k), device="cuda", dtype=torch.float16, generator=g) * 2.0 - 1.0

            amax_a = a.float().abs().amax()
            sf_g_a = float((FP4_AMAX * FP8_AMAX) / amax_a)
            sf_g_a_t = torch.tensor([sf_g_a], device=a.device, dtype=torch.float32)

            amax_x = x.float().abs().amax()
            sf_g_x = float((FP4_AMAX * FP8_AMAX) / amax_x)
            sf_g_x_t = torch.tensor([sf_g_x], device=x.device, dtype=torch.float32)

            q_a, scale_a = _nvfp4_quantize(a, sf_g_a_t)
            q_x, scale_x = _nvfp4_quantize(x, sf_g_x_t)
            return q_a, scale_a, q_x, scale_x, sf_g_a, sf_g_x

        return {"name": name, "dims": (m, k), "create_inputs": create_inputs}

    def generate_test_cases(self) -> List[Dict[str, Any]]:
        configs = [
            (2048, 2048, "2048 x 2048"),
            (4096, 2048, "4096 x 2048"),
            (8192, 4096, "8192 x 4096"),
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
