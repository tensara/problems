import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem
from tolerances import tol_for

class conv_1d(Problem):
    """1D convolution problem."""

    numeric_category = "GEMM_CONV"
    
    def __init__(self):
        super().__init__(
            name="conv-1d"
        )
    
    def reference_solution(self, input_signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of 1D convolution.
        
        Args:
            input_signal: Input signal tensor of shape (N)
            kernel: Convolution kernel tensor of shape (K)
            
        Returns:
            Result of convolution with zero padding
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            # Ensure kernel size is odd
            assert kernel.size(0) % 2 == 1, "Kernel size must be odd"
            
            # Perform 1D convolution using PyTorch's built-in function
            # Convert to shape expected by conv1d: [batch, channels, length]
            input_reshaped = input_signal.view(1, 1, -1)
            kernel_reshaped = kernel.view(1, 1, -1)
            
            # Calculate padding size to maintain the same output size
            padding = kernel.size(0) // 2
            
            # Perform convolution
            result = torch.nn.functional.conv1d(
                input_reshaped, 
                kernel_reshaped, 
                padding=padding
            )
            
            # Reshape back to original dimensions
            return result.view(-1)
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for 1D convolution.
        
        Returns:
            List of test case dictionaries with varying sizes
        """
        
        test_configs = [
            (65536, 8191),
            (32768, 8191),
            (131072, 8191),
            (524288, 8191)
        ]
        
        return [
            {
                "name": f"N={signal_size}, K={kernel_size}",
                "signal_size": signal_size,
                "kernel_size": kernel_size,
                "create_inputs": lambda s=signal_size, k=kernel_size: (
                    torch.rand(s, device="cuda", dtype=dtype) * 10.0 - 5.0,
                    torch.rand(k, device="cuda", dtype=dtype) * 2.0 - 1.0
                )
            }
            for signal_size, kernel_size in test_configs
        ]
    
    def generate_sample(self, dtype: torch.dtype = torch.float32) -> List[Dict[str, Any]]:
        """
        Generate a single sample test case for debugging or interactive runs.
        
        Returns:
            A list containing a single test case dictionary
        """
        signal_size, kernel_size = (16, 3)
        return {
            "name": f"N={signal_size}, K={kernel_size}",
            "signal_size": signal_size,
            "kernel_size": kernel_size,
            "create_inputs": lambda s=signal_size, k=kernel_size: (
                torch.arange(1, s + 1, device="cuda", dtype=dtype).float(), 
                torch.tensor([1.0, 2.0, 1.0], device="cuda", dtype=dtype)
            )
        }
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the convolution result is correct.
        
        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            
        Returns:
            Tuple of (is_correct, debug_info)
        """
        tol = tol_for(dtype, self.numeric_category)
        if tol is None:
            is_close = torch.equal(actual_output, expected_output)
        else:
            is_close = torch.allclose(actual_output, expected_output, rtol=tol.rtol, atol=tol.atol)
        
        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            max_diff = torch.max(torch.abs(diff)).item()
            mean_diff = torch.mean(torch.abs(diff)).item()
            
            # Find indices of largest differences
            _, top_indices = torch.topk(torch.abs(diff), min(5, diff.numel()))
            
            sample_diffs = {
                f"{idx.item()}": {
                    "expected": expected_output[idx].item(),
                    "actual": actual_output[idx].item(),
                    "diff": diff[idx].item()
                }
                for idx in top_indices
            }
            
            debug_info = {
                "max_difference": max_diff,
                "mean_difference": mean_diff,
                "sample_differences": sample_diffs
            }
        
        return is_close, debug_info
    
    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the 1D convolution solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # input_signal
                ctypes.POINTER(ctypes.c_float),  # kernel
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_size_t,                 # signal_size (N)
                ctypes.c_size_t                  # kernel_size (K)
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
        # For each output element, we do kernel_size multiplications and kernel_size-1 additions
        # Each output element requires K multiplications and K-1 additions
        N = test_case["signal_size"]
        K = test_case["kernel_size"]
        

        # mult-adds count as 2 FLOPs
        flops_per_element = 2 * K - 1  
        
        return N * flops_per_element
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the signal size N and kernel size K
        """
        N = test_case["signal_size"]
        K = test_case["kernel_size"]
        return [N, K]
