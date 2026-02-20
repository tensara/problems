import torch
from typing import List, Dict, Tuple, Any

from problem import Problem

class leaky_relu(Problem):
    """Leaky ReLU activation function problem."""
    
    is_exact = False
    
    parameters = [
        {"name": "input",  "type": "float",  "pointer": True,  "const": True},
        {"name": "alpha",  "type": "float",  "pointer": False, "const": False},
        {"name": "output", "type": "float",  "pointer": True,  "const": False},
        {"name": "n",      "type": "size_t", "pointer": False, "const": False},
        {"name": "m",      "type": "size_t", "pointer": False, "const": False},
    ]

    def __init__(self):
        super().__init__(
            name="leaky-relu"
        )
    
    def reference_solution(self, input_matrix: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        PyTorch implementation of Leaky ReLU.
        
        Args:
            input_matrix: Input matrix of shape (M, N)
            alpha: Slope for negative values
            
        Returns:
            Result of Leaky ReLU activation
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=input_matrix.dtype):
            return torch.nn.functional.leaky_relu(input_matrix, alpha)

    def generate_test_cases(self) -> List[Dict[str, Any]]:
        dtype = self.param_dtype(0)

        matrix_sizes = [
            (4096, 4096),
            (6144, 4096)
        ]
        
        alpha_values = [0.01, 0.05, 0.1, 0.2]
        
        test_cases = []
        for m, n in matrix_sizes:
            for alpha in alpha_values:
                name = f"{m}x{n}, alpha={alpha}"
                seed = Problem.get_seed(f"{self.name}_{name}_{(m, n)}")
                test_cases.append({
                    "name": name,
                    "rows": m,
                    "cols": n,
                    "alpha": alpha,
                    "create_inputs": lambda m=m, n=n, alpha=alpha, seed=seed, dtype=dtype: (
                        *(lambda g: (
                            torch.rand((m, n), device="cuda", dtype=dtype, generator=g) * 2.0 - 1.0,  # uniform [-1, 1]
                        ))(torch.Generator(device="cuda").manual_seed(seed)),
                        alpha,
                    )
                })
        
        return test_cases

    def generate_sample(self) -> List[Dict[str, Any]]:
        dtype = self.param_dtype(0)
        m, n = (4, 4)
        alpha = 0.01
        return {
            "name": f"Sample ({m}x{n}), alpha={alpha}",
            "rows": m,
            "cols": n,
            "alpha": alpha,
            "create_inputs": lambda m=m, n=n, alpha=alpha: (
                torch.tensor([[-2.0, -1.0, 0.0, 1.0],
                                [2.0, -0.5, 0.5, -1.5],
                                [1.5, 0.0, -2.5, 3.0],
                                [-3.0, 2.5, -0.2, 0.4]], device="cuda", dtype=dtype),
                alpha
            )
        }

    def verify_result(self, expected_output: torch.Tensor,
                     actual_output: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        is_close = torch.allclose(actual_output, expected_output, rtol=1e-4, atol=1e-6)
        
        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            max_diff = torch.max(torch.abs(diff)).item()
            mean_diff = torch.mean(torch.abs(diff)).item()
            
            # Find indices of largest differences
            flat_diff = diff.flatten()
            _, top_indices = torch.topk(torch.abs(flat_diff), min(5, flat_diff.numel()))
            
            # Convert flat indices back to 2D coordinates
            m, n = expected_output.shape
            sample_diffs = {}
            for i, idx in enumerate(top_indices):
                row = idx.item() // n
                col = idx.item() % n
                sample_diffs[f"({row}, {col})"] = {
                    "expected": expected_output[row, col].item(),
                    "actual": actual_output[row, col].item(),
                    "diff": diff[row, col].item()
                }
                        
            debug_info = {
                "max_difference": max_diff,
                "mean_difference": mean_diff,
                "sample_differences": sample_diffs
            }
        
        return is_close, debug_info

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the number of floating point operations for the problem.
        
        Args:
            test_case: The test case dictionary

        IMPORTANT: Comments are required. Outline the FLOPs calculation.
            
        Returns:
            Number of floating point operations
        """
        M = test_case["rows"]
        N = test_case["cols"]
        
        # M*N FLOPs:
        # - Each element requires 1 comparison and at most 1 multiplication
        # - We count this as 1 FLOP per element as per the test case
        return M * N
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the rows M, columns N, and alpha value
        """
        M = test_case["rows"]
        N = test_case["cols"]
        return [M, N]