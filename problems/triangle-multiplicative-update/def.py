import math

import torch
from typing import Any, Dict, List, Tuple

from problem import Problem


class triangle_multiplicative_update(Problem):
    """Masked outgoing triangle multiplicative update."""

    is_exact = False

    parameters = [
        {"name": "left", "type": "float", "pointer": True, "const": True},
        {"name": "right", "type": "float", "pointer": True, "const": True},
        {"name": "mask", "type": "float", "pointer": True, "const": True},
        {"name": "output", "type": "float", "pointer": True, "const": False},
        {"name": "B", "type": "size_t", "pointer": False, "const": False},
        {"name": "N", "type": "size_t", "pointer": False, "const": False},
        {"name": "H", "type": "size_t", "pointer": False, "const": False},
    ]

    def __init__(self):
        super().__init__(name="triangle-multiplicative-update")

    def reference_solution(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reference masked outgoing triangle multiplicative update.

        Args:
            left: Tensor of shape (B, N, N, H), indexed as left[b, i, k, h].
            right: Tensor of shape (B, N, N, H), indexed as right[b, j, k, h].
            mask: Tensor of shape (B, N, N), with 0/1 entries applied to both paths.

        Returns:
            Tensor of shape (B, N, N, H), where
            output[b, i, j, h] = sum_k left[b, i, k, h] * mask[b, i, k]
                                * right[b, j, k, h] * mask[b, j, k].
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=left.dtype):
            mask_expanded = mask.unsqueeze(-1)
            left_masked = left * mask_expanded
            right_masked = right * mask_expanded

            b, n, _, h = left.shape
            left_bh = left_masked.permute(0, 3, 1, 2).reshape(b * h, n, n)
            right_bh_t = right_masked.permute(0, 3, 2, 1).reshape(b * h, n, n)
            out_bh = torch.bmm(left_bh, right_bh_t)
            return out_bh.reshape(b, h, n, n).permute(0, 2, 3, 1).contiguous()

    def _make_case(
        self,
        *,
        b: int,
        n: int,
        h: int,
        distribution: str,
        mask_mode: str,
    ) -> Dict[str, Any]:
        name = f"B={b}, N={n}, H={h}, {distribution}, {mask_mode}"
        seed = Problem.get_seed(f"{self.name}_{name}")
        dtype = self.param_dtype(0)

        def create_inputs(
            b: int = b,
            n: int = n,
            h: int = h,
            distribution: str = distribution,
            mask_mode: str = mask_mode,
            seed: int = seed,
            dtype: torch.dtype = dtype,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            generator = torch.Generator(device="cuda").manual_seed(seed)
            shape = (b, n, n, h)

            if distribution == "cauchy":
                # Inverse-CDF sampling keeps the case deterministic under the
                # per-test CUDA generator.
                left_u = torch.rand(shape, device="cuda", dtype=torch.float32, generator=generator)
                right_u = torch.rand(shape, device="cuda", dtype=torch.float32, generator=generator)
                # Clamp and scale to keep the reference finite while preserving
                # heavy-tailed inputs.
                left = (2.0 * torch.tan(math.pi * (left_u - 0.5))).clamp(-16.0, 16.0).to(dtype) * 0.125
                right = (2.0 * torch.tan(math.pi * (right_u - 0.5))).clamp(-16.0, 16.0).to(dtype) * 0.125
            else:
                left = torch.randn(shape, device="cuda", dtype=dtype, generator=generator) * 0.5
                right = torch.randn(shape, device="cuda", dtype=dtype, generator=generator) * 0.5

            if mask_mode == "nomask":
                mask = torch.ones((b, n, n), device="cuda", dtype=dtype)
            else:
                mask = torch.randint(
                    0,
                    2,
                    (b, n, n),
                    device="cuda",
                    dtype=torch.int32,
                    generator=generator,
                ).to(dtype)

            return left.contiguous(), right.contiguous(), mask.contiguous()

        return {
            "name": name,
            "B": b,
            "N": n,
            "H": h,
            "distribution": distribution,
            "mask_mode": mask_mode,
            "create_inputs": create_inputs,
        }

    def generate_test_cases(self) -> List[Dict[str, Any]]:
        return [
            self._make_case(b=1, n=128, h=64, distribution="normal", mask_mode="nomask"),
            self._make_case(b=1, n=256, h=128, distribution="normal", mask_mode="random-mask"),
            self._make_case(b=2, n=256, h=128, distribution="cauchy", mask_mode="random-mask"),
            self._make_case(b=1, n=512, h=128, distribution="normal", mask_mode="nomask"),
            self._make_case(b=1, n=1024, h=128, distribution="normal", mask_mode="random-mask"),
        ]

    def generate_sample(self) -> Dict[str, Any]:
        dtype = self.param_dtype(0)
        b, n, h = 1, 4, 2

        return {
            "name": "Sample B=1, N=4, H=2",
            "B": b,
            "N": n,
            "H": h,
            "distribution": "deterministic",
            "mask_mode": "random-mask",
            "create_inputs": lambda: (
                torch.arange(1, b * n * n * h + 1, device="cuda", dtype=dtype).view(b, n, n, h) / 16.0,
                torch.arange(1, b * n * n * h + 1, device="cuda", dtype=dtype).flip(0).view(b, n, n, h) / 32.0,
                torch.tensor(
                    [[[1, 1, 0, 1], [1, 0, 1, 1], [1, 1, 1, 0], [0, 1, 1, 1]]],
                    device="cuda",
                    dtype=dtype,
                ),
            ),
        }

    def verify_result(
        self,
        expected_output: torch.Tensor,
        actual_output: torch.Tensor,
    ) -> Tuple[bool, Dict[str, Any]]:
        is_close = torch.allclose(actual_output, expected_output, rtol=2e-2, atol=2e-2)

        debug_info: Dict[str, Any] = {}
        if not is_close:
            diff = actual_output.float() - expected_output.float()
            abs_diff = torch.abs(diff)
            flat_diff = abs_diff.flatten()
            _, top_indices = torch.topk(flat_diff, min(5, flat_diff.numel()))
            shape = expected_output.shape

            sample_differences = {}
            for flat_idx in top_indices:
                idx = []
                value = flat_idx.item()
                for dim_size in reversed(shape):
                    idx.insert(0, value % dim_size)
                    value //= dim_size
                idx_tuple = tuple(idx)
                sample_differences[str(idx_tuple)] = {
                    "expected": expected_output[idx_tuple].item(),
                    "actual": actual_output[idx_tuple].item(),
                    "diff": diff[idx_tuple].item(),
                }

            debug_info = {
                "max_difference": flat_diff.max().item(),
                "mean_difference": abs_diff.mean().item(),
                "sample_differences": sample_differences,
            }

        return is_close, debug_info

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        FLOPs for the masked outgoing triangle update.

        For each output element output[b, i, j, h], the reduction over k uses
        one multiply for left*right and one add into the accumulator. There are
        B*N*N*H output elements and N reduction terms, so the contraction costs
        approximately 2*B*N^3*H FLOPs. The mask multiplications add
        2*B*N^2*H FLOPs, one for each path element.
        """
        b = test_case["B"]
        n = test_case["N"]
        h = test_case["H"]
        return int((2 * b * n * n * n * h) + (2 * b * n * n * h))

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        return [test_case["B"], test_case["N"], test_case["H"]]
