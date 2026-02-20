import torch
from typing import List, Dict, Tuple, Any

from problem import Problem

class gelu(Problem):
    """GELU activation function problem."""
    
    is_exact = False

    parameters = [
        {"name": "input", "type": "float", "pointer": True, "const": True},
        {"name": "output", "type": "float", "pointer": True, "const": False},
        {"name": "n", "type": "size_t", "pointer": False, "const": False},
        {"name": "m", "type": "size_t", "pointer": False, "const": False},
    ]

    
    def __init__(self):
        super().__init__(
            name="gelu"
        )
    
    def reference_solution(self, input_matrix: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of GELU.
        
        Args:
            input_matrix: Input matrix of shape (M, N)
            
        Returns:
            Result of GELU activation
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=input_matrix.dtype):
            return torch.nn.functional.gelu(input_matrix, approximate="tanh")
    
    def generate_test_cases(self) -> List[Dict[str, Any]]:
        """
        Generate test cases for GELU.
        
        Returns:
            List of test case dictionaries with varying sizes
        """
        dtype = self.param_dtype(0)

        # Test case configurations with specific matrix sizes
        test_configs = [
            ("4096x4096", 4096, 4096),
            ("6144x4096", 6144, 4096),
            ("4096x7168", 4096, 7168),
            ("4096x8192", 4096, 8192),
            ("8192x8192", 8192, 8192)
        ]
        
        test_cases = []
        for name, m, n in test_configs:
            seed = Problem.get_seed(f"{self.name}_{name}_{(m, n)}")
            test_cases.append({
                "name": name,
                "rows": m,
                "cols": n,
                "create_inputs": lambda m=m, n=n, seed=seed, dtype=dtype: (
                    *(lambda g: (
                        torch.rand((m, n), device="cuda", dtype=dtype, generator=g) * 2.0 - 1.0,  # uniform [-1, 1]
                    ))(torch.Generator(device="cuda").manual_seed(seed)),
                )
            })
        return test_cases
    
    def generate_sample(self) -> List[Dict[str, Any]]:
        """
        Generate a single sample test case for debugging or interactive runs.
        
        Returns:
            A list containing a single test case dictionary
        """
        dtype = self.param_dtype(0)

        m, n = (8, 8) # Sample dimensions
        return {
            "name": f"Sample ({m}x{n})",
            "rows": m,
            "cols": n,
            "create_inputs": lambda m=m, n=n: (
                torch.linspace(-5, 5, m * n, device="cuda", dtype=dtype).view(m, n),
            )
        }
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the GELU result is correct.
        
        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            
        Returns:
            Tuple of (is_correct, debug_info)
        """
        is_close = torch.allclose(actual_output, expected_output, rtol=1e-4, atol=2e-5)
        
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
        # Extract dimensions from test case
        M = test_case["rows"]
        N = test_case["cols"]
        
        # For the approximation: 0.5x * (1 + tanh(sqrt(2/π) * (x + 0.044715x^3)))
        # Operations per element:
        # - 1 multiply for x^3 (x * x * x): 3 FLOPs
        # - 1 multiply for 0.044715x^3: 1 FLOP
        # - 1 addition for (x + 0.044715x^3): 1 FLOP
        # - 1 multiply for sqrt(2/π) * (x + 0.044715x^3): 1 FLOP
        # - 1 tanh operation: ~15 FLOPs (approximate cost of transcendental function)
        # - Used this https://stackoverflow.com/a/43021821 to approximate the cost of tanh
        # - 1 addition for (1 + tanh(...)): 1 FLOP
        # - 1 multiply for 0.5x: 1 FLOP
        # - 1 multiply for 0.5x * (1 + tanh(...)): 1 FLOP
        # Total: approximately 24 FLOPs per element
        
        flops_per_element = 24
        return M * N * flops_per_element
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the rows M and columns N
        """
        M = test_case["rows"]
        N = test_case["cols"]
        return [M, N]