import torch
from typing import List, Dict, Tuple, Any

from problem import Problem

class hinge_loss(Problem):
    """Hinge Loss problem for binary classification."""
    
    is_exact = False

    parameters = [
        {"name": "predictions", "type": "float", "pointer": True, "const": True},
        {"name": "targets", "type": "float", "pointer": True, "const": True},
        {"name": "output", "type": "float", "pointer": True, "const": False},
        {"name": "n", "type": "size_t", "pointer": False, "const": False},
    ]

    
    def __init__(self):
        super().__init__(
            name="hinge-loss"
        )
    
    def reference_solution(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of element-wise Hinge Loss.
        
        Args:
            predictions: Predictions tensor of shape (N,)
            targets: Binary targets tensor of shape (N,) with values in {-1, 1}
            
        Returns:
            Element-wise hinge loss tensor of shape (N,)
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=predictions.dtype):
            return torch.clamp(1 - predictions * targets, min=0)
    
    def generate_test_cases(self) -> List[Dict[str, Any]]:
        """
        Generate test cases for Hinge Loss.
        
        Returns:
            List of test case dictionaries with varying sizes.
        """
        dtype = self.param_dtype(0)

        tensor_sizes = [
            1048576,      # 1M elements
            4194304,      # 4M elements
            16777216,     # 16M elements
            67108864,     # 64M elements
        ]
        
        test_cases = []
        for n in tensor_sizes:
            seed = Problem.get_seed(f"{self.name}_N={n}")
            test_cases.append({
                "name": f"N={n}",
                "n": n,
                "create_inputs": lambda n=n, seed=seed, dtype=dtype: (
                    (lambda g: (
                        torch.randn(n, device="cuda", dtype=dtype, generator=g),           # predictions
                        torch.randint(0, 2, (n,), device="cuda", dtype=dtype, generator=g) * 2 - 1  # targets in {-1, 1}
                    ))(torch.Generator(device="cuda").manual_seed(seed))
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

        n = 8 
        return {
            "name": f"Sample N={n}",
            "n": n,
            "create_inputs": lambda n=n: (
                torch.tensor([-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, -0.5], device="cuda", dtype=dtype),
                torch.tensor([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0], device="cuda", dtype=dtype)
            )
        }
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the Hinge Loss result is correct.
        
        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            
        Returns:
            Tuple of (is_correct, debug_info)
        """

        is_close = torch.allclose(actual_output, expected_output, rtol=1e-4, atol=1e-4)
        
        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            max_diff = torch.max(torch.abs(diff)).item()
            mean_diff = torch.mean(torch.abs(diff)).item()
            
            # Find indices of largest differences
            flat_diff = diff.flatten()
            _, top_indices = torch.topk(torch.abs(flat_diff), min(5, flat_diff.numel()))
            
            n = expected_output.numel()
            sample_diffs = {}
            for i, idx in enumerate(top_indices):
                sample_diffs[f"{idx.item()}"] = {
                    "expected": expected_output.flatten()[idx].item(),
                    "actual": actual_output.flatten()[idx].item(),
                    "diff": diff.flatten()[idx].item()
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
        N = test_case["n"]
        
        # N * 3 FLOPs: For each element:
        # 1. mult = predictions[i] * targets[i] (1 FLOP)
        # 2. sub = 1 - mult                     (1 FLOP)
        # 3. max(0, sub)                        (1 FLOP, comparison)
        return N * 3
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the number of elements N.
        """
        N = test_case["n"]
        return [N]
