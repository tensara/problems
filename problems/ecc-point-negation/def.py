import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem

class ecc_point_negation(Problem):
    """
    Batched ECC point negation over F_p with p = 2^61 - 1 (fits in uint64_t).

      Input : xs[i], ys[i], p
      Output: out[i, 0] = xs[i]
              out[i, 1] = (p - ys[i]) % p

    Notes:
      - We do NOT enforce on-curve membership. This is a field-level warm-up.
      - Output is a single (N, 2) tensor to match a single output buffer in CUDA.
    """

    def __init__(self):
        super().__init__(name="ecc-point-negation")
        self.p: int = (1 << 61) - 1  # 2^61 - 1 (Mersenne prime)

    # ---------------------------
    # Reference (PyTorch)
    # ---------------------------
    def reference_solution(self, xs: torch.Tensor, ys: torch.Tensor, p: int) -> torch.Tensor:
        with torch.no_grad():
            assert xs.shape == ys.shape
            assert xs.dtype == torch.int64 and ys.dtype == torch.int64
            device = xs.device
            p64 = torch.tensor(p, dtype=torch.int64, device=device)

            assert xs.dtype == torch.int64 and ys.dtype == torch.int64
            assert xs.device.type == "cuda" and ys.device.type == "cuda"

            neg_y = (p64 - (ys % p64)) % p64
            out = torch.stack((xs, neg_y), dim=1)
            return out
 

    # ---------------------------
    # Test-case generation
    # ---------------------------
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate increasingly large batches to stress memory bandwidth.
        """
        sizes = [
            ("N = 262,144",   262_144),
            ("N = 524,288",   524_288),
            ("N = 1,048,576", 1_048_576),
            ("N = 2,097,152", 2_097_152),
        ]

        test_cases = []
        for name, N in sizes:
            seed = Problem.get_seed(f"{self.name}_{name}_{(N,)}")

            test_cases.append({
                "name": name,
                "dims": (N,),  # batch length
                "create_inputs": lambda N=N, seed=seed, p=self.p, create_inputs=self._create_inputs: (
                    (lambda g: (
                        (lambda xs, ys: (xs, ys, int(p)))(*create_inputs(N, g))
                    ))(torch.Generator(device="cuda").manual_seed(seed))
                )
            })
        return test_cases

    def _create_inputs(self, N: int, generator: torch.Generator):
        """
        Create random field elements modulo p as int64 (to match uint64_t).
        """
        device = "cuda"
        p = self.p
        xs = torch.randint(low=0, high=p, size=(N,), device=device, dtype=torch.int64, generator=generator)
        ys = torch.randint(low=0, high=p, size=(N,), device=device, dtype=torch.int64, generator=generator)
        return xs, ys

    # ---------------------------
    # Sample (small) for debugging
    # ---------------------------
    def generate_sample(self, dtype: torch.dtype = torch.int64) -> Dict[str, Any]:
        name = "Sample (N = 8)"
        device = "cuda"
        p = self.p
        assert isinstance(p, int), f"type(p)={type(p)}"
        assert p == (1 << 61) - 1, f"p corrupted: {p}"
    

        xs = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], device=device, dtype=torch.int64)
        ys = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=device, dtype=torch.int64)

        assert xs.dtype == torch.int64 and ys.dtype == torch.int64
        assert xs.device.type == "cuda" and ys.device.type == "cuda"

        def create_sample_inputs():
            return (xs, ys, int(p))
 

        return {
            "name": name,
            "dims": (xs.numel(),),
            "create_inputs": create_sample_inputs
        }

    # ---------------------------
    # Verification
    # ---------------------------
    def verify_result(
        self,
        expected_output: torch.Tensor,   # shape (N, 2), int64
        actual_output: torch.Tensor,     # shape (N, 2), int64 (from single output buffer)
        dtype: torch.dtype
    ) -> Tuple[bool, Dict[str, Any]]:
        # strict equality — integer ops
        ok = torch.equal(expected_output, actual_output)

        debug: Dict[str, Any] = {}
        if not ok:
            diff = (expected_output != actual_output)
            # first row with any mismatch
            bad_rows = torch.where(diff.any(dim=1))[0]
            if bad_rows.numel() > 0:
                i = int(bad_rows[0].item())
                debug = {
                    "index": i,
                    "expected": [int(expected_output[i, 0].item()), int(expected_output[i, 1].item())],
                    "actual": [int(actual_output[i, 0].item()), int(actual_output[i, 1].item())],
                }
            else:
                debug = {"message": "Mismatch detected but no row isolated."}
        return ok, debug

    # ---------------------------
    # C ABI for CUDA solution
    # ---------------------------
    def get_function_signature(self) -> Dict[str, Any]:
        """
        CUDA signature (single interleaved output: length = 2*N):

          void solution(
              const uint64_t* xs,     // length N
              const uint64_t* ys,     // length N
              const uint64_t  p,      // scalar
              uint64_t*       out_xy, // length 2*N (pairs: x, neg_y)
              size_t          N
          );
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_uint64),  # xs
                ctypes.POINTER(ctypes.c_uint64),  # ys
                ctypes.c_uint64,                  # p
                ctypes.POINTER(ctypes.c_uint64),  # out_xy
                ctypes.c_size_t,                  # N
            ],
            "restype": None,
        }

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Count integer-ops equivalent.

        Each element:
          - 1 subtraction (p - y)
          - 1 modular reduction (% p)
        Total = 2 ops per element
        """
        N = test_case["dims"][0]
        return 2 * N

    def get_mem(self, test_case: Dict[str, Any]) -> int:
        """
        Get the memory usage for the problem. Assumed to be all in DRAM
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            Memory usage in bytes
        """
        N = test_case["dims"][0]
        
        # Input: xs (N) + ys (N) - both uint64_t (8 bytes)
        # Output: out_xy (2*N) - uint64_t (8 bytes)
        dtype_bytes = 8  # 8 bytes per uint64_t element
        return (N + N + 2 * N) * dtype_bytes

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Provide N as the trailing scalar param to the CUDA solution.
        """
        N = test_case["dims"][0]
        return [N]
