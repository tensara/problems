import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class conv2d_relu_hardswish(Problem):
    """2D convolution followed by ReLU followed by HardSwish activation fusion problem."""
    
    def __init__(self):
        super().__init__(
            name="conv2d-relu-hardswish"
        )
    
    def reference_solution(self, input_image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of 2D convolution followed by ReLU followed by HardSwish.
        
        Args:
            input_image: Input image tensor of shape (H, W)
            kernel: Convolution kernel tensor of shape (Kh, Kw)
            
        Returns:
            Result of conv2d -> ReLU -> HardSwish fusion
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            # Ensure kernel sizes are odd
            assert kernel.size(0) % 2 == 1, "Kernel height must be odd"
            assert kernel.size(1) % 2 == 1, "Kernel width must be odd"
            
            # Perform 2D convolution using PyTorch's built-in function
            # Convert to shape expected by conv2d: [batch, channels, height, width]
            input_reshaped = input_image.view(1, 1, input_image.size(0), input_image.size(1))
            kernel_reshaped = kernel.view(1, 1, kernel.size(0), kernel.size(1))
            
            # Calculate padding size to maintain the same output size
            padding_h = kernel.size(0) // 2
            padding_w = kernel.size(1) // 2
            
            # Perform convolution
            conv_result = torch.nn.functional.conv2d(
                input_reshaped, 
                kernel_reshaped, 
                padding=(padding_h, padding_w)
            )
            
            # Reshape back to original dimensions
            conv_result = conv_result.view(input_image.size(0), input_image.size(1))
            
            # Apply ReLU activation
            relu_result = torch.nn.functional.relu(conv_result)
            
            # Apply HardSwish activation: x * ReLU6(x + 3) / 6
            hardswish_result = relu_result * torch.nn.functional.relu6(relu_result + 3) / 6
            
            return hardswish_result
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for conv2d-relu-hardswish fusion.
        
        Returns:
            List of test case dictionaries with varying sizes
        """
        test_configs = [
            (512, 512, 3, 3),
            (1024, 1024, 5, 5),
            (2048, 2048, 7, 7),
            (512, 512, 9, 9),
            (1024, 1024, 11, 11),
            (2048, 2048, 13, 13)
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
    
    def generate_sample(self, dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
        """
        Generate a single sample test case for debugging or interactive runs.
        
        Returns:
            A test case dictionary
        """
        h, w, kh, kw = (4, 4, 3, 3) # Sample configuration (kernel dims must be odd)
        return {
            "name": f"H={h}, W={w}, Kh={kh}, Kw={kw}",
            "height": h,
            "width": w,
            "kernel_height": kh,
            "kernel_width": kw,
            "create_inputs": lambda h=h, w=w, kh=kh, kw=kw: (
                torch.tensor([
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0]
                ], device="cuda", dtype=dtype),
                torch.tensor([
                    [1.0, 0.0, -1.0],
                    [2.0, 0.0, -2.0],
                    [1.0, 0.0, -1.0]
                ], device="cuda", dtype=dtype)
            )
        }
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the conv2d-relu-hardswish result is correct.
        
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
        Get the function signature for the conv2d-relu-hardswish solution.
        
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
            
        Returns:
            Number of floating point operations
        """
        # Conv2D FLOPS = 2 * H * W * Kh * Kw
        # ReLU FLOPS = 0 (comparison operation only)
        # HardSwish FLOPS = 5 * H * W (add, relu6, mul, div, mul)
        H = test_case["height"]
        W = test_case["width"]
        Kh = test_case["kernel_height"]
        Kw = test_case["kernel_width"]
        
        conv_flops = 2 * H * W * Kh * Kw
        relu_flops = 0
        hardswish_flops = 5 * H * W
        
        return conv_flops + relu_flops + hardswish_flops
    
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