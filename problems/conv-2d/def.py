import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem
from .solution import Conv2dSolutions

class conv_2d(Conv2dSolutions, Problem):
    """2D convolution problem."""
    
    def __init__(self):
        super().__init__(
            name="conv-2d"
        )
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for 2D convolution.
        
        Returns:
            List of test case dictionaries with varying sizes
        """
        test_configs = [
            (4096, 4096, 9, 9),
            (8192, 8192, 11, 11),
            (16384, 16384, 13, 13),
            (1024, 1024, 31, 31),
            (2048, 2048, 63, 63),
            (4096, 4096, 127, 127)
        ]
        
        return [
            {
                "name": f"H={h}, W={w}, Kh={kh}, Kw={kw}",
                "height": h,
                "width": w,
                "kernel_height": kh,
                "kernel_width": kw,
                "create_inputs": lambda h=h, w=w, kh=kh, kw=kw: (
                    torch.rand((h, w), device="cuda", dtype=dtype) * 10.0 - 5.0,  # uniform [-5, 5]
                    torch.rand((kh, kw), device="cuda", dtype=dtype) * 2.0 - 1.0  # uniform [-1, 1]
                )
            }
            for h, w, kh, kw in test_configs
        ]
    
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
            h, w = expected_output.shape
            sample_diffs = {}
            for i, idx in enumerate(top_indices):
                row = idx.item() // w
                col = idx.item() % w
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
        Get the function signature for the 2D convolution solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # input_image
                ctypes.POINTER(ctypes.c_float),  # kernel
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_size_t,                 # height (H)
                ctypes.c_size_t,                 # width (W)
                ctypes.c_size_t,                 # kernel_height (Kh)
                ctypes.c_size_t                  # kernel_width (Kw)
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
        # For 2D convolution, each output pixel requires Kh*Kw multiplications and Kh*Kw-1 additions
        H = test_case["height"]
        W = test_case["width"]
        Kh = test_case["kernel_height"]
        Kw = test_case["kernel_width"]
        
        # Total FLOPs for the entire image: H*W output pixels, each requiring:
        # Kh*Kw multiplications + (Kh*Kw-1) additions = 2*Kh*Kw - 1 FLOPs
        # Following the test case's flop calculation which uses 2*H*W*Kh*Kw
        # This is slightly different from our detailed calculation but aligns with the test code
        return 2 * H * W * Kh * Kw
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the image height H, width W, kernel height Kh, and kernel width Kw
        """
        H = test_case["height"]
        W = test_case["width"]
        Kh = test_case["kernel_height"]
        Kw = test_case["kernel_width"]
        return [H, W, Kh, Kw]