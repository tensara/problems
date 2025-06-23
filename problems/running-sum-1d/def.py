import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem

class running_sum_1d(Problem):
    """1D running sum problem with fix sized sliding window. """
    
    def __init__(self):
        super().__init__(
            name="running-sum-1d"
        )
    
    def reference_solution(self, input_tensor: torch.Tensor, window_size: int) -> torch.Tensor:
        """
        PyTorch implementation of 1D running sum problem.
        
        Args:
            input_tensor: Input tensor
            window_size: Size of the sliding window
            
        Returns:
            Sums of the input tensor over the sliding window
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            
            # Perform 1D convolution using PyTorch's built-in function
            # using kernel of ones to compute the running sum
            input_reshaped = input_tensor.view(1, 1, -1)
            kernel = torch.ones(window_size, dtype=input_tensor.dtype, device=input_tensor.device)
            kernel_reshaped = kernel.view(1, 1, -1)
            
            # Calculate padding size to maintain the same output size
            padding = window_size // 2
            
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
        Generate test cases for 1D running sum problem.
        
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
                "name": f"N={signal_size}, W={window_size}",
                "signal_size": signal_size,
                "window_size": window_size,
                "create_inputs": lambda s=signal_size, w=window_size: (
                    torch.rand(s, device="cuda", dtype=dtype) * 10.0 - 5.0,
                    w
                )
            }
            for signal_size, window_size in test_configs
        ]
    
    def generate_sample(self, dtype: torch.dtype = torch.float32) -> List[Dict[str, Any]]:
        """
        Generate a single sample test case for debugging or interactive runs.
        
        Returns:
            A list containing a single test case dictionary
        """
        signal_size, window_size = (16, 3)
        return {
            "name": f"N={signal_size}, W={window_size}",
            "signal_size": signal_size,
            "window_size": window_size,
            "create_inputs": lambda s=signal_size, w=window_size: (
                torch.arange(1, s + 1, device="cuda", dtype=dtype).float(),
                w
            )
        }
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the test result is correct.
        
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
        Get the function signature for the 1D running sum problem solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # input_signal
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_size_t,                 # signal_size (N)
                ctypes.c_size_t                  # window_size (W)
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
        # Calculation of precfix sum of input data requires N addition operations
        # Each output element requires one substruction of two elements of prefix sum
        N = test_case["signal_size"]
        
        return 2*N 
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the signal size N and window_size W
        """
        N = test_case["signal_size"]
        W = test_case["window_size"]
        return [N, W]