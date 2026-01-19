import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class matrix_vector(Problem):
    """Matrix vector multiplication problem."""
    
    is_exact = False
    
    def __init__(self):
        super().__init__(
            name="matrix-vector-multiplication"
        )
    
    def reference_solution(self, matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of matrix-vector multiplication.
        
        Args:
            matrix: Input matrix of shape (M, K)
            vector: Input vector of shape (K)
            
        Returns:
            Result of matrix-vector multiplication of shape (M)
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=matrix.dtype):
            return torch.matmul(matrix, vector)
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for matrix-vector multiplication.
        
        Returns:
            List of test case dictionaries with varying sizes
        """
        # Test case configurations with specific matrix and vector sizes
        test_configs = [
            (4096, 4096),
            (6144, 4096),
            (7168, 4096),
            (8192, 4096),
            (9216, 4096)
        ]
        
        test_cases = []
        for m, k in test_configs:
            seed = Problem.get_seed(f"{self.name}_M={m}_K={k}")
            test_cases.append({
                "name": f"M={m}, K={k}",
                "rows": m,
                "cols": k,
                "create_inputs": lambda m=m, k=k, seed=seed, dtype=dtype: (
                    *(lambda g: (
                        torch.rand((m, k), device="cuda", dtype=dtype, generator=g) * 2 - 1,  # uniform [-1, 1]
                        torch.rand((k), device="cuda", dtype=dtype, generator=g) * 2 - 1      # uniform [-1, 1]
                    ))(torch.Generator(device="cuda").manual_seed(seed)),
                )
            })
        return test_cases
    
    def generate_sample(self, dtype: torch.dtype = torch.float32) -> List[Dict[str, Any]]:
        """
        Generate sample test cases for matrix-vector multiplication with predictable inputs.

        Returns:
            List of sample test case dictionaries.
        """
        shape = (8, 8)  # Sample 2D tensor shape
        return {
            "name": f"shape={shape}",
            "rows": 8,
            "cols": 8,
            "create_inputs": lambda shape=shape: (
                torch.arange(1, torch.prod(torch.tensor(shape)).item() + 1, device="cuda", dtype=dtype).float().view(*shape),
                torch.arange(1, shape[1] + 1, device="cuda", dtype=dtype).float()
            )
        }

    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the matrix-vector multiplication result is correct.
        
        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            
        Returns:
            Tuple of (is_correct, debug_info)
        """
        is_close = torch.allclose(actual_output, expected_output, rtol=1e-4, atol=1e-3)
        
        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            max_diff = torch.max(torch.abs(diff)).item()
            mean_diff = torch.mean(torch.abs(diff)).item()
            
            # Find indices of largest differences
            _, top_indices = torch.topk(torch.abs(diff), min(5, diff.numel()))
            
            sample_diffs = {}
            for i, idx in enumerate(top_indices):
                sample_diffs[f"{idx.item()}"] = {
                    "expected": expected_output[idx].item(),
                    "actual": actual_output[idx].item(),
                    "diff": diff[idx].item()
                }
            
            debug_info = {
                "max_difference": max_diff,
                "mean_difference": mean_diff,
                "sample_differences": sample_diffs
            }
        
        return is_close, debug_info
    
    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the matrix-vector multiplication solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # matrix
                ctypes.POINTER(ctypes.c_float),  # vector
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_size_t,                 # rows (M)
                ctypes.c_size_t                  # columns (K)
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
        # Extract dimensions from test case
        M = test_case["rows"]
        K = test_case["cols"]
        
        # M*K*2 FLOPs:
        # - Each output element requires K MAD operations
        # - Each MAD (Multiply-Add) counts as 2 FLOPs
        return M * K * 2
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the rows M and columns K
        """
        M = test_case["rows"]
        K = test_case["cols"]
        return [M, K]