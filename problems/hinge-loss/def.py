import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class hinge_loss(Problem):
    """Hinge Loss problem for binary classification."""
    
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
        with torch.no_grad():
            return torch.clamp(1 - predictions * targets, min=0)
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for Hinge Loss.
        
        Returns:
            List of test case dictionaries with varying sizes.
        """
        tensor_sizes = [
            1048576,      # 1M elements
            4194304,      # 4M elements
            16777216,     # 16M elements
            67108864,     # 64M elements
        ]
        
        test_cases = []
        for n in tensor_sizes:
            test_cases.append({
                "name": f"N={n}",
                "n": n,
                "create_inputs": lambda n=n: (
                    torch.randn(n, device="cuda", dtype=dtype),           # predictions
                    torch.randint(0, 2, (n,), device="cuda", dtype=dtype) * 2 - 1  # targets in {-1, 1}
                )
            })
        
        return test_cases
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the Hinge Loss result is correct.
        
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
    
    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the Hinge Loss solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # predictions
                ctypes.POINTER(ctypes.c_float),  # targets
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_size_t                  # n (number of elements)
            ],
            "restype": None
        }
    
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
