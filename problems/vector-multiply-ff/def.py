import torch
from typing import List, Dict, Tuple, Any

from problem import Problem

# Mersenne prime p = 2^31 - 1
P = (1 << 31) - 1  # 2147483647

class vector_multiply_ff(Problem):
    """Vector multiplication over F_p with p = 2^31 - 1 (Mersenne)."""

    is_exact = True

    parameters = [
        {"name": "d_input1", "type": "uint32_t", "pointer": True, "const": True},
        {"name": "d_input2", "type": "uint32_t", "pointer": True, "const": True},
        {"name": "d_output", "type": "uint32_t", "pointer": True, "const": False},
        {"name": "n", "type": "size_t", "pointer": False, "const": False},
    ]

    def __init__(self):
        super().__init__(name="vector-multiply-ff")

    # ---------------------------
    # Reference solution
    # ---------------------------
    def reference_solution(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Element-wise multiplication modulo p for uint32 inputs.
        Returns uint32 on CUDA.
        """
        with torch.no_grad():
            # Promote to 64-bit to avoid overflow on the product, then reduce mod P.
            prod = (A.to(torch.int64) * B.to(torch.int64)) % P
            return prod.to(torch.uint32).to("cuda")

    # ---------------------------
    # Test-case generation
    # ---------------------------
    def generate_test_cases(self) -> List[Dict[str, Any]]:
        dtype = self.param_dtype(0)

        sizes = [
            ("n = 2^20", 1_048_576),
            ("n = 2^22", 4_194_304),
            ("n = 2^24", 16_777_216),
            ("n = 2^25", 33_554_432),
        ]
        tests: List[Dict[str, Any]] = []
        for name, n in sizes:
            seed = Problem.get_seed(f"{self.name}_{name}")
            def make_inputs(n=n, seed=seed):
                g = torch.Generator(device="cuda").manual_seed(seed)
                a = torch.randint(0, P, (n,), device="cuda", dtype=torch.uint32, generator=g)
                b = torch.randint(0, P, (n,), device="cuda", dtype=torch.uint32, generator=g)
                return (a, b)
            tests.append({"name": name, "dims": (n,), "create_inputs": make_inputs})
        return tests

    # ---------------------------
    # Sample (small)
    # ---------------------------
    def generate_sample(self) -> Dict[str, Any]:
        dtype = self.param_dtype(0)

        name = "Sample (n = 8)"
        size = 8
        return {
            "name": name,
            "dims": (size,),
            "create_inputs": lambda size=size: (
                torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], device="cuda", dtype=torch.uint32),
                torch.tensor([8, 7, 6, 5, 4, 3, 2, 1], device="cuda", dtype=torch.uint32),
            ),
        }

    # ---------------------------
    # Verification
    # ---------------------------
    def verify_result(
        self,
        expected_output: torch.Tensor,
        actual_output: torch.Tensor,
    ) -> Tuple[bool, Dict[str, Any]]:
        exp = expected_output.to(torch.uint32)
        act = actual_output.to(torch.uint32)
        is_equal = torch.equal(exp, act)
        debug_info: Dict[str, Any] = {}
        if not is_equal:
            diff_mask = (exp != act)
            idxs = torch.where(diff_mask)[0]
            if idxs.numel() > 0:
                i = int(idxs[0].item())
                debug_info = {
                    "index": i,
                    "expected": int(exp[i].item()),
                    "actual": int(act[i].item()),
                    "num_mismatches": int(diff_mask.sum().item()),
                }
        return is_equal, debug_info

    # ---------------------------
    # C ABI
    # ---------------------------
    def get_flops(self, test_case: Dict[str, Any]) -> int:
        # One multiply + one modular reduction per element (approximate as 2 ops)
        return 2 * test_case["dims"][0]

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        return [test_case["dims"][0]]
