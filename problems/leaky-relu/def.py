import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem
from .solution import LeakyReluSolutions


class leaky_relu(LeakyReluSolutions, Problem):
    """Leaky ReLU activation function problem."""
    
    def __init__(self):
        super().__init__(
            name="leaky-relu"
        )
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for Leaky ReLU.
        
        Returns:
            List of test case dictionaries with varying sizes and alpha values
        """
        # Test case configurations with specific matrix sizes and alpha values
        matrix_sizes = [
            (4096, 4096),
            (6144, 4096)
        ]
        
        alpha_values = [0.01, 0.05, 0.1, 0.2]
        
        test_cases = []
        for m, n in matrix_sizes:
            for alpha in alpha_values:
                test_cases.append({
                    "name": f"{m}x{n}, alpha={alpha}",
                    "rows": m,
                    "cols": n,
                    "alpha": alpha,
                    "create_inputs": lambda m=m, n=n, alpha=alpha: (
                        torch.rand((m, n), device="cuda", dtype=dtype) * 10.0 - 5.0,  # uniform [-5, 5]
                        alpha
                    )
                })
        
        return test_cases
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the Leaky ReLU result is correct.
        
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
    
    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the Leaky ReLU solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # input_matrix
                ctypes.c_float,                  # alpha
                ctypes.POINTER(ctypes.c_float),  # output_matrix
                ctypes.c_size_t,                 # rows (M)
                ctypes.c_size_t                  # columns (N)
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