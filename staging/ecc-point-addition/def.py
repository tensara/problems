import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem

class ecc_point_addition(Problem):
    """
    Batched ECC point addition over F_p with p = 2^61 - 1 (fits in uint64_t).

      Input : xs1[i], ys1[i], xs2[i], ys2[i], p
      Output: out[i, 0] = x3
              out[i, 1] = y3

    Notes:
      - We assume inputs never lead to the point at infinity.
      - That means denominators in slope formulas are always invertible mod p.
    """

    def __init__(self):
        super().__init__(name="ecc-point-addition")
        self.p: int = (1 << 61) - 1  # 2^61 - 1 (Mersenne prime)

    # ---------------------------
    # Modular arithmetic
    # ---------------------------
    @staticmethod
    def _mod(x: torch.Tensor, p: int) -> torch.Tensor:
        return x.remainder(p)

    @staticmethod
    def _mod_mul(a: torch.Tensor, b: torch.Tensor, p: int) -> torch.Tensor:
        return (a * b).remainder(p)

    @staticmethod
    def _mod_inv(a: torch.Tensor, p: int) -> torch.Tensor:
        """
        Modular inverse using Fermat's little theorem:
          a^(p-2) mod p
        Works since p is prime.
        """
        device = a.device
        res = torch.ones_like(a)
        base = a.remainder(p)
        exp = p - 2
        while exp > 0:
            if exp & 1:
                res = (res * base).remainder(p)
            base = (base * base).remainder(p)
            exp >>= 1
        return res

    # ---------------------------
    # Reference solution
    # ---------------------------
    def reference_solution(
        self,
        xs1: torch.Tensor,
        ys1: torch.Tensor,
        xs2: torch.Tensor,
        ys2: torch.Tensor,
        p: int
    ) -> torch.Tensor:
        with torch.no_grad():
            assert xs1.shape == ys1.shape == xs2.shape == ys2.shape
            N = xs1.numel()

            x1, y1 = xs1 % p, ys1 % p
            x2, y2 = xs2 % p, ys2 % p

            same_point = (x1 == x2) & (y1 == y2)

            # slope λ
            lam = torch.empty_like(x1)
            lam[~same_point] = self._mod_mul(
                (y2 - y1)[~same_point],
                self._mod_inv((x2 - x1)[~same_point], p),
                p,
            )
            lam[same_point] = self._mod_mul(
                3 * x1[same_point] * x1[same_point] % p,
                self._mod_inv((2 * y1[same_point]) % p, p),
                p,
            )

            x3 = (lam * lam - x1 - x2).remainder(p)
            y3 = (lam * (x1 - x3) - y1).remainder(p)

            out = torch.stack((x3, y3), dim=1)
            return out

    # ---------------------------
    # Test-case generation
    # ---------------------------
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        sizes = [
            ("N = 262,144",   262_144),
            ("N = 524,288",   524_288),
            ("N = 1,048,576", 1_048_576),
            ("N = 2,097,152", 2_097_152),
        ]
        test_cases = []
        for name, N in sizes:
            xs1, ys1, xs2, ys2 = self._create_inputs(N)
            def create_inputs_closure(_xs1=xs1, _ys1=ys1, _xs2=xs2, _ys2=ys2, _p=self.p):
                return (_xs1, _ys1, _xs2, _ys2, int(_p))
            test_cases.append({
                "name": name,
                "dims": (N,),
                "create_inputs": create_inputs_closure,
            })
        return test_cases

    def _create_inputs(self, N: int):
        device = "cuda"
        p = self.p
        xs1 = torch.randint(1, p, (N,), device=device, dtype=torch.int64)
        ys1 = torch.randint(1, p, (N,), device=device, dtype=torch.int64)
        xs2 = torch.randint(1, p, (N,), device=device, dtype=torch.int64)
        ys2 = torch.randint(1, p, (N,), device=device, dtype=torch.int64)
        return xs1, ys1, xs2, ys2

    # ---------------------------
    # Sample (small)
    # ---------------------------
    def generate_sample(self, dtype: torch.dtype = torch.int64) -> Dict[str, Any]:
        p = self.p
        device = "cuda"
        xs1 = torch.tensor([1, 2, 3, 4], device=device, dtype=torch.int64)
        ys1 = torch.tensor([2, 3, 4, 5], device=device, dtype=torch.int64)
        xs2 = torch.tensor([5, 6, 7, 8], device=device, dtype=torch.int64)
        ys2 = torch.tensor([6, 7, 8, 9], device=device, dtype=torch.int64)
        def create_inputs_closure():
            return (xs1, ys1, xs2, ys2, int(p))
        return {
            "name": "Sample (N=4)",
            "dims": (4,),
            "create_inputs": create_inputs_closure,
        }

    # ---------------------------
    # Verification
    # ---------------------------
    def verify_result(
        self,
        expected_output: torch.Tensor,
        actual_output: torch.Tensor,
        dtype: torch.dtype
    ) -> Tuple[bool, Dict[str, Any]]:
        ok = torch.equal(expected_output, actual_output)
        debug = {}
        if not ok:
            diff = (expected_output != actual_output)
            bad_rows = torch.where(diff.any(dim=1))[0]
            if bad_rows.numel() > 0:
                i = int(bad_rows[0].item())
                debug = {
                    "index": i,
                    "expected": expected_output[i].tolist(),
                    "actual": actual_output[i].tolist(),
                }
        return ok, debug

    # ---------------------------
    # C ABI for CUDA solution
    # ---------------------------
    def get_function_signature(self) -> Dict[str, Any]:
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_uint64),  # xs1
                ctypes.POINTER(ctypes.c_uint64),  # ys1
                ctypes.POINTER(ctypes.c_uint64),  # xs2
                ctypes.POINTER(ctypes.c_uint64),  # ys2
                ctypes.c_uint64,                  # p
                ctypes.POINTER(ctypes.c_uint64),  # out_xy
                ctypes.c_size_t,                  # N
            ],
            "restype": None,
        }

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        N = test_case["dims"][0]
        return 12 * N

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        return [test_case["dims"][0]]
