import torch
from typing import List, Dict, Tuple, Any

from problem import Problem

BLOCK_SIZE = 32


class mxfp8_gemv(Problem):

    is_exact = False

    parameters = [
        {"name": "q_a", "type": "uint8_t", "pointer": True, "const": True},
        {"name": "scale_a", "type": "uint8_t", "pointer": True, "const": True},
        {"name": "q_x", "type": "uint8_t", "pointer": True, "const": True},
        {"name": "scale_x", "type": "uint8_t", "pointer": True, "const": True},
        {"name": "y", "type": "float", "pointer": True, "const": False},
        {"name": "m", "type": "size_t", "pointer": False, "const": False},
        {"name": "k", "type": "size_t", "pointer": False, "const": False},
    ]

    def __init__(self):
        super().__init__(name="mxfp8-gemv")

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

    def _dequantize(self, q_bytes: torch.Tensor, scale_bytes: torch.Tensor) -> torch.Tensor:
        _, to_dtype = self._mx_tensor_api()
        q_lp = q_bytes.view(torch.float8_e4m3fn)
        scale_e8m0 = scale_bytes.view(torch.float8_e8m0fnu)
        return to_dtype(q_lp, scale_e8m0, torch.float8_e4m3fn, BLOCK_SIZE, torch.float32).float()

    def reference_solution(
        self, q_a: torch.Tensor, scale_a: torch.Tensor, q_x: torch.Tensor, scale_x: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            a_deq = self._dequantize(q_a, scale_a)
            x_deq = self._dequantize(q_x, scale_x)
            m, k = a_deq.shape

            assert a_deq.shape == (m, k)
            assert x_deq.shape == (1, k)

            return torch.matmul(a_deq, x_deq.squeeze(0))

    def _make_case(self, m: int, k: int, name: str) -> Dict[str, Any]:
        if k % BLOCK_SIZE != 0:
            raise ValueError(f"K must be divisible by {BLOCK_SIZE}, got K={k}")

        seed = Problem.get_seed(f"{self.name}_{name}_M={m}_K={k}")

        def create_inputs(m=m, k=k, seed=seed):
            to_mx, _ = self._mx_tensor_api()

            g = torch.Generator(device="cuda").manual_seed(seed)
            a = torch.randn((m, k), device="cuda", dtype=torch.float32, generator=g)
            x = torch.randn((k, 1), device="cuda", dtype=torch.float32, generator=g)

            scale_a_e8m0, a_lp = to_mx(a, torch.float8_e4m3fn, BLOCK_SIZE)
            scale_x_e8m0, x_lp = to_mx(x, torch.float8_e4m3fn, BLOCK_SIZE)

            print("shape of scale_x_e8m0:", scale_x_e8m0.shape)
            print("shape of x_lp:", x_lp.shape)

            q_a = a_lp.contiguous().view(torch.uint8)
            scale_a = scale_a_e8m0.contiguous().view(torch.uint8)
            q_x = x_lp.contiguous().view(torch.uint8)
            scale_x = scale_x_e8m0.contiguous().view(torch.uint8)

            return q_a, scale_a, q_x, scale_x

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
