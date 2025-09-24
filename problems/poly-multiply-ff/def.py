
import torch
import ctypes
from typing import List, Dict, Tuple, Any
from problem import Problem

# Mersenne prime p = 2^31 - 1
P = (1 << 31) - 1  # 2147483647

class poly_multiply_ff(Problem):
    """Polynomial multiplication over F_p with p = 2^31 - 1 (Mersenne), using uint32 coefficients."""

    def __init__(self):
        super().__init__(name="poly-multiply-ff")

    # ---------------------------
    # Reference (CPU, exact math)
    # ---------------------------
    def reference_solution(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Naive O(n^2) convolution over Z_p. Works on CPU with Python ints, returns CUDA uint32.
        """
        MOD = P
        with torch.no_grad():
            A_cpu = A.detach().to("cpu").contiguous().view(-1).tolist()
            B_cpu = B.detach().to("cpu").contiguous().view(-1).tolist()
            n = len(A_cpu)
            assert n == len(B_cpu)

            # ensure [0, P) and Python-int arithmetic
            a = [int(x) % MOD for x in A_cpu]
            b = [int(x) % MOD for x in B_cpu]

            out_len = 2 * n - 1
            c = [0] * out_len
            for i in range(n):
                ai = a[i]
                for j in range(n):
                    c[i + j] = (c[i + j] + (ai * b[j]) % MOD) % MOD

            # Build on CPU as uint32, then move to CUDA
            c_tensor = torch.tensor(c, dtype=torch.uint32)
            return c_tensor.to("cuda")

    # ---------------------------
    # Test-case generation
    # ---------------------------
    def generate_test_cases(self, dtype: torch.dtype = torch.uint32) -> List[Dict[str, Any]]:
        sizes = [("n = 2^6", 64), ("n = 2^8", 256), ("n = 2^10", 1024)]
        tests: List[Dict[str, Any]] = []
        for name, n in sizes:
            def make_inputs(n=n):
                # Generate directly on CUDA in [0, P)
                A = torch.randint(0, P, (n,), dtype=torch.uint32, device="cuda")
                B = torch.randint(0, P, (n,), dtype=torch.uint32, device="cuda")
                return (A, B)
            tests.append({"name": name, "dims": (n,), "create_inputs": make_inputs})
        return tests

    # ---------------------------
    # Sample (small)
    # ---------------------------
    def generate_sample(self, dtype: torch.dtype = torch.uint32) -> Dict[str, Any]:
        def make_sample():
            A = torch.tensor([1, 2, 3, 4], dtype=torch.uint32, device="cuda")
            B = torch.tensor([4, 3, 2, 1], dtype=torch.uint32, device="cuda")
            return (A, B)
        return {"name": "Sample (n = 4)", "dims": (4,), "create_inputs": make_sample}

    # ---------------------------
    # Verification
    # ---------------------------
    def verify_result(
        self,
        expected_output: torch.Tensor,
        actual_output: torch.Tensor,
        dtype: torch.dtype,
    ) -> Tuple[bool, Dict[str, Any]]:
        # Compare as uint32 (already mod P)
        exp = expected_output.to(torch.uint32)
        act = actual_output.to(torch.uint32)
        ok = torch.equal(exp, act)
        debug: Dict[str, Any] = {}
        if not ok:
            diffmask = (exp != act)
            idxs = torch.where(diffmask)[0]
            if idxs.numel() > 0:
                i = int(idxs[0].item())
                debug = {
                    "index": i,
                    "expected": int(exp[i].item()),
                    "actual": int(act[i].item()),
                    "mismatch_count": int(diffmask.sum().item()),
                }
        return ok, debug

    # ---------------------------
    # CUDA C ABI
    # ---------------------------
    def get_function_signature(self) -> Dict[str, Any]:
        """
        void solution(
            const uint32_t* A,   // length n
            const uint32_t* B,   // length n
            uint32_t*       C,   // length (2*n - 1)
            size_t          n
        );
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_uint32),  # A
                ctypes.POINTER(ctypes.c_uint32),  # B
                ctypes.POINTER(ctypes.c_uint32),  # C
                ctypes.c_size_t,                  # n
            ],
            "restype": None,
        }

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        n = test_case["dims"][0]
        return 2 * n * n  # one mul + one add per term

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        return [test_case["dims"][0]]
