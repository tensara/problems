import torch
from typing import List, Dict, Tuple, Any

from problem import Problem

class dropout(Problem):
    """Dropout regularization problem."""
    
    is_exact = False

    parameters = [
        {"name": "input", "type": "float", "pointer": True, "const": True},
        {"name": "mask", "type": "float", "pointer": True, "const": True},
        {"name": "p", "type": "float", "pointer": False, "const": False},
        {"name": "output", "type": "float", "pointer": True, "const": False},
        {"name": "n", "type": "size_t", "pointer": False, "const": False},
        {"name": "m", "type": "size_t", "pointer": False, "const": False},
    ]

    
    def __init__(self):
        super().__init__(
            name="dropout"
        )

    def reference_solution(self, input_matrix: torch.Tensor, mask: torch.Tensor, p: float) -> torch.Tensor:
        """
        PyTorch implementation of Dropout with a mask.
        
        Args:
            input_matrix: Input matrix of shape (M, N)
            mask: Binary mask of shape (M, N) containing 0s and 1s
            p: Dropout probability
            
        Returns:
            Result of dropout operation.
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=input_matrix.dtype):
            scale = 1.0 / (1.0 - p) if p < 1.0 else 0.0
            return (input_matrix * mask) * scale

    def generate_test_cases(self) -> List[Dict[str, Any]]:
        """
        Generate test cases for Dropout.
        
        Returns:
            List of test case dictionaries with varying sizes and probabilities.
        """
        dtype = self.param_dtype(0)

        """test cases"""
        test_configs = [
            ("4096x4096_p05", 4096, 4096, 0.5),
            ("6144x4096_p02", 6144, 4096, 0.2),
            ("4096x7168_p08", 4096, 7168, 0.8),
            ("4096x8192_p01", 4096, 8192, 0.1),
            ("8192x8192_p05", 8192, 8192, 0.5)
        ]
        
        test_cases = []
        for name, m, n, p in test_configs:
            seed = Problem.get_seed(f"{self.name}_{name}_{(m, n, p)}")
            test_cases.append({
                "name": name,
                "rows": m,
                "cols": n,
                "p": p,
                "create_inputs": lambda m=m, n=n, p=p, seed=seed, dtype=dtype: (
                    *(lambda g: (
                        torch.randn((m, n), device="cuda", dtype=dtype, generator=g), 
                        (torch.rand((m, n), device="cuda", dtype=dtype, generator=g) > p).to(dtype), 
                    ))(torch.Generator(device="cuda").manual_seed(seed)),
                    p,
                )
            })
        return test_cases
    
    def generate_sample(self) -> List[Dict[str, Any]]:
        """
        A single sample test case for debugging or interactive runs.
        
        Returns:
            A list containing a single test case dictionary.
        """
        dtype = self.param_dtype(0)

        m, n, p = (4, 4, 0.5)
        return {
            "name": f"Sample ({m}x{n})",
            "rows": m,
            "cols": n,
            "p": p,
            "create_inputs": lambda m=m, n=n, p=p: (
                torch.tensor([
                    [-5.0, -2.5, 0.0, 2.5],
                    [-4.0, -1.5, 1.0, 3.5],
                    [-3.0, -0.5, 2.0, 4.5],
                    [-2.0,  0.5, 3.0, 5.0]
                ], device="cuda", dtype=dtype),
                torch.tensor([
                    [1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0]
                ], device="cuda", dtype=dtype),
                p,
            )
        }

    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the Dropout result is correct.
        
        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            
        Returns:
            Tuple of (is_correct, debug_info)
        """
        is_close = torch.allclose(actual_output, expected_output, rtol=1e-5, atol=1e-5)
        
        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            max_diff = torch.max(torch.abs(diff)).item()
            mean_diff = torch.mean(torch.abs(diff)).item()
            
            # indices of largest differences
            flat_diff = diff.flatten()
            _, top_indices = torch.topk(torch.abs(flat_diff), min(5, flat_diff.numel()))
            
            # flat indices to 2D coordinates
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
            
        Returns:
            Number of floating point operations
        """
       
        M = test_case["rows"]
        N = test_case["cols"]
        
        return 2 * M * N
    
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
