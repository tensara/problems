import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class box_blur(Problem):
    """Box blur problem."""
    
    def __init__(self):
        super().__init__(
            name="box-blur"
        )
    
    def reference_solution(self, input_image: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """
        PyTorch implementation of box blur.
        
        Args:
            input_image: Input grayscale image of shape (height, width)
            kernel_size: Size of the blur kernel (must be odd, e.g., 3, 5, 7)
            
        Returns:
            Blurred image of shape (height, width)
        """
        with torch.no_grad():
            h, w = input_image.shape
            pad = kernel_size // 2
            
            padded = torch.nn.functional.pad(input_image, (pad, pad, pad, pad), mode='constant', value=0)
            input_reshaped = padded.view(1, 1, h + 2*pad, w + 2*pad)
            
            kernel = torch.ones(1, 1, kernel_size, kernel_size, device=input_image.device)
            
            y = torch.arange(h, device=input_image.device)
            x = torch.arange(w, device=input_image.device)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            
            i_start = torch.clamp(yy - pad, 0, h)
            i_end = torch.clamp(yy + pad + 1, 0, h) 
            j_start = torch.clamp(xx - pad, 0, w)
            j_end = torch.clamp(xx + pad + 1, 0, w)
            
            valid_elements = (i_end - i_start) * (j_end - j_start)
            
            output = torch.nn.functional.conv2d(input_reshaped, kernel).view(h, w)
            output = output / valid_elements
            
            return output
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for box blur.
        
        Returns:
            List of test case dictionaries with varying image sizes and kernel sizes
        """
        image_sizes = [
            (1920, 1080),
            (2048, 2048)
        ]
        
        kernel_sizes = [15, 21, 27]
        
        test_cases = []
        for height, width in image_sizes:
            for kernel_size in kernel_sizes:
                seed = Problem.get_seed(f"{self.name}_H={height}_W={width}_K={kernel_size}")
                test_cases.append({
                    "name": f"{height}x{width}, kernel={kernel_size}x{kernel_size}",
                    "height": height,
                    "width": width,
                    "kernel_size": kernel_size,
                    "create_inputs": lambda h=height, w=width, k=kernel_size, seed=seed, dtype=dtype: (
                        *(lambda g: (
                            torch.rand((h, w), device="cuda", dtype=dtype, generator=g) * 255.0,
                        ))(torch.Generator(device="cuda").manual_seed(seed)),
                        k
                    )
                })
        
        return test_cases
    
    def generate_sample(self, dtype: torch.dtype = torch.float32) -> List[Dict[str, Any]]:
        """
        Generate a single sample test case for debugging or interactive runs.
        
        Returns:
            A list containing a single test case dictionary
        """
        image_size = (8, 8)
        kernel_size = 3
        return {
            "name": "Sample (h=8, w=8, kernel=3x3)",
            "height": image_size[0],
            "width": image_size[1],
            "kernel_size": kernel_size,
            "create_inputs": lambda h=image_size[0], w=image_size[1], k=kernel_size: (
                torch.rand((h, w), device="cuda", dtype=dtype) * 255.0,
                k
            )
        }
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the box blur result is correct.
        
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
        Get the function signature for the box blur solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # input_image
                ctypes.c_int,                    # kernel_size
                ctypes.POINTER(ctypes.c_float),  # output_image
                ctypes.c_size_t,                 # height
                ctypes.c_size_t                  # width
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
        height = test_case["height"]
        width = test_case["width"]
        kernel_size = test_case["kernel_size"]
        
        # For each output pixel:
        # - We access up to kernel_size x kernel_size input pixels
        # - We sum them (kernel_size^2 - 1 additions)
        # - We divide by the count to get average (1 division)
        # Total FLOPs per pixel: kernel_size^2 additions + 1 division
        flops_per_pixel = kernel_size * kernel_size
        return height * width * flops_per_pixel
    
    def get_mem(self, test_case: Dict[str, Any]) -> int:
        """
        Get the memory usage for the problem. Assumed to be all in DRAM
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            Memory usage in bytes
        """
        height = test_case["height"]
        width = test_case["width"]
        kernel_size = test_case["kernel_size"]
        
        # Input image: height*width elements
        # Kernel: kernel_size*kernel_size elements (but this is typically small and reused)
        # Output: height*width elements
        # For memory bandwidth, we count input + output (kernel is small and cached)
        dtype_bytes = 4  # 4 bytes per float32 element
        return (height * width + height * width) * dtype_bytes
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the height, width, and kernel size
        """
        height = test_case["height"]
        width = test_case["width"]
        return [height, width] 