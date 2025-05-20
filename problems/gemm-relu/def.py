import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem
from .solution import GemmReluSolutions


class gemm_relu(GemmReluSolutions, Problem):
    """GEMM with Bias and ReLU problem."""
    
    def __init__(self):
        super().__init__(
            name="gemm-relu"
        )
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for GEMM with Bias and ReLU.
        
        Returns:
            List of test case dictionaries with varying sizes
        """
        
        test_configs = [
            (512, 6144, 1024),
            (512, 8192, 1024),
            (512, 8192, 2048),
            (1024, 2048, 2048),
            (1024, 4096, 2048),
            (1024, 4096, 4096)
        ]
        
        return [
            {
                "name": f"B={batch_size}, N={in_features}, M={out_features}",
                "batch_size": batch_size,
                "in_features": in_features,
                "out_features": out_features,
                "create_inputs": lambda b=batch_size, n=in_features, m=out_features: (
                    torch.rand((b, n), device="cuda", dtype=dtype) * 2 - 1,
                    torch.rand((m, n), device="cuda", dtype=dtype) * 2 - 1,  
                    torch.rand((m), device="cuda", dtype=dtype) * 2 - 1  
                )
            }
            for batch_size, in_features, out_features in test_configs
        ]
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the GEMM with Bias and ReLU result is correct.
        
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
            flat_diff = diff.flatten()
            _, top_indices = torch.topk(torch.abs(flat_diff), min(5, flat_diff.numel()))
            
            # Convert flat indices back to 2D coordinates
            B, M = expected_output.shape
            sample_diffs = {}
            for i, idx in enumerate(top_indices):
                row = idx.item() // M
                col = idx.item() % M
                sample_diffs[f"({row}, {col})"] = {
                    "expected": expected_output[row, col].item(),
                    "actual": actual_output[row, col].item(),
                    "diff": diff[row, col].item()
                }
            
            # Check for differences in activation pattern
            expected_zeros = (expected_output == 0).sum().item()
            actual_zeros = (actual_output == 0).sum().item()
            
            debug_info = {
                "max_difference": max_diff,
                "mean_difference": mean_diff,
                "sample_differences": sample_diffs,
                "message": f'Expected {expected_zeros} zeros, got {actual_zeros} zeros'
            }
        
        return is_close, debug_info
    
    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the GEMM with Bias and ReLU solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # input_matrix
                ctypes.POINTER(ctypes.c_float),  # weights
                ctypes.POINTER(ctypes.c_float),  # bias
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_size_t,                 # batch_size (B)
                ctypes.c_size_t,                 # input_features (N)
                ctypes.c_size_t                  # output_features (M)
            ],
            "restype": None
        }
    
    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the number of floating point operations for the problem.

        IMPORTANT: Comments are required. Outline the FLOPs calculation.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            Number of floating point operations
        """
        
        B = test_case["batch_size"]
        N = test_case["in_features"]
        M = test_case["out_features"]
        
        # 2*B*N*M FLOPs for matrix multiplication:
        # - Each output element requires N MAD operations (2*N FLOPs)
        # - There are B*M output elements
        #
        # B*M FLOPs for bias addition:
        # - Each of the B*M output elements requires 1 addition
        return 2 * B * N * M + B * M
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the batch_size B, input_features N, and output_features M
        """
        B = test_case["batch_size"]
        N = test_case["in_features"]
        M = test_case["out_features"]
        return [B, N, M]