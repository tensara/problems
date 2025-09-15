
import torch
import ctypes
from typing import List, Dict, Tuple, Any
from problem import Problem

# Goldilocks prime
P = (1 << 64) - (1 << 32) + 1  # 18446744069414584321

class poly_multiply_goldilocks(Problem):
    """Polynomial multiplication over the Goldilocks field (p = 2^64 - 2^32 + 1)."""

    def __init__(self):
        super().__init__(name="poly-multiply-ff")

    # ---------------------------
    # Reference (CPU, exact math)
    # ---------------------------
    def reference_solution(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Naive O(n^2) convolution over Z_p.
        We compute on CPU using Python bigints for exactness, then move to CUDA as uint64.
        """
        MODULUS = P
        with torch.no_grad():
            # Move to CPU and ensure Python-int semantics
            A_cpu = A.detach().to("cpu").contiguous()
            B_cpu = B.detach().to("cpu").contiguous()
            n = int(A_cpu.numel())
            assert n == int(B_cpu.numel())

            # Convert to Python ints to avoid overflow issues
            a_list = [int(x) % MODULUS for x in A_cpu.view(-1).tolist()]
            b_list = [int(x) % MODULUS for x in B_cpu.view(-1).tolist()]

            out_len = 2 * n - 1
            c = [0] * out_len
            for i in range(n):
                ai = a_list[i]
                for j in range(n):
                    c[i + j] = (c[i + j] + (ai * b_list[j]) % MODULUS) % MODULUS

            # Pack into torch and return to GPU as uint64
            c_tensor = torch.tensor(c, dtype=torch.uint64)
            return c_tensor.to("cuda")

    # ---------------------------
    # Test-case generation
    # ---------------------------
    def generate_test_cases(self, dtype: torch.dtype = torch.uint64) -> List[Dict[str, Any]]:
        # Sizes chosen as powers of two (friendly for later NTT variants)
        sizes = [("n = 2^6", 64), ("n = 2^8", 256), ("n = 2^10", 1024)]
        tests = []
        for name, n in sizes:
            def make_inputs(n=n):
                # Generate on CPU using Python ints in [0, P), then move to CUDA as uint64
                A_cpu = torch.randint(0, 2**63, (n,), dtype=torch.int64)  # wide-ish
                B_cpu = torch.randint(0, 2**63, (n,), dtype=torch.int64)
                # Reduce mod P using Python, then cast to uint64
                A_list = [int(x) % P for x in A_cpu.tolist()]
                B_list = [int(x) % P for x in B_cpu.tolist()]
                A = torch.tensor(A_list, dtype=torch.uint64, device="cuda")
                B = torch.tensor(B_list, dtype=torch.uint64, device="cuda")
                return (A, B)
            tests.append({"name": name, "dims": (n,), "create_inputs": make_inputs})
        return tests

    # ---------------------------
    # Sample (small)
    # ---------------------------
    def generate_sample(self, dtype: torch.dtype = torch.uint64) -> Dict[str, Any]:
        def make_sample():
            A = torch.tensor([1, 2, 3, 4], dtype=torch.uint64, device="cuda")
            B = torch.tensor([4, 3, 2, 1], dtype=torch.uint64, device="cuda")
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
        ok = torch.equal(expected_output, actual_output)
        debug = {}
        if not ok:
            # Show first mismatch and a quick diff summary
            diff = (expected_output != actual_output)
            idxs = torch.where(diff)[0]
            if idxs.numel() > 0:
                i = int(idxs[0].item())
                debug["index"] = i
                debug["expected"] = int(expected_output[i].item())
                debug["actual"] = int(actual_output[i].item())
            debug["mismatch_count"] = int(diff.sum().item())
        return ok, debug

    # ---------------------------
    # CUDA C ABI
    # ---------------------------
    def get_function_signature(self) -> Dict[str, Any]:
        """
        void solution(
            const uint64_t* A,     // length n
            const uint64_t* B,     // length n
            uint64_t*       C,     // length (2*n - 1)
            size_t          n
        );
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_uint64),  # A
                ctypes.POINTER(ctypes.c_uint64),  # B
                ctypes.POINTER(ctypes.c_uint64),  # C
                ctypes.c_size_t,                  # n
            ],
            "restype": None,
        }

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        n = test_case["dims"][0]
        # ~1 mul + 1 add per inner loop
        return 2 * n * n

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        return [test_case["dims"][0]]

