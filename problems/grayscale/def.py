import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class grayscale(Problem):
    """Grayscale conversion problem."""
    
    def __init__(self):
        super().__init__(
            name="grayscale"
        )
    
    def reference_solution(self, rgb_image: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of RGB to grayscale conversion.
        
        Args:
            rgb_image: Input RGB image of shape (height, width, 3)
            
        Returns:
            Grayscale image of shape (height, width)
        """
        with torch.no_grad():
            # Apply standard RGB to grayscale conversion weights
            # RGB image is in shape (height, width, 3)
            r, g, b = rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]
            grayscale = 0.299 * r + 0.587 * g + 0.114 * b
            return grayscale
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for grayscale conversion.
        
        Returns:
            List of test case dictionaries with varying image sizes
        """
        image_sizes = [
            (512, 512),    
            (1024, 768),    
            (1920, 1080),   
            (3840, 2160)    
        ]
        
        test_cases = []
        for height, width in image_sizes:
            test_cases.append({
                "name": f"{height}x{width}",
                "height": height,
                "width": width,
                "channels": 3,
                "create_inputs": lambda h=height, w=width: (
                    torch.rand((h, w, 3), device="cuda", dtype=dtype) * 255.0,
                )
            })
        
        return test_cases
    
    def generate_sample(self, dtype: torch.dtype = torch.float32) -> List[Dict[str, Any]]:
        """
        Generate a single sample test case for debugging or interactive runs.
        
        Returns:
            A list containing a single test case dictionary
        """
        name = "Sample (h=8, w=8)"
        height = 8
        width = 8
        return {
            "name": name,
            "height": height,
            "width": width,
            "channels": 3,
            "create_inputs": lambda h=height, w=width: (
                torch.rand((h, w, 3), device="cuda", dtype=dtype) * 255.0,
            )
        }

    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the grayscale conversion result is correct.
        
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
        Get the function signature for the grayscale conversion solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # rgb_image
                ctypes.POINTER(ctypes.c_float),  # grayscale_output
                ctypes.c_size_t,                 # height
                ctypes.c_size_t,                 # width
                ctypes.c_size_t                  # channels
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
        
        # For each pixel:
        # - 3 multiplications (one for each channel)
        # - 2 additions (to sum the weighted values)
        # Total: 5 FLOPs per pixel
        return height * width * 5
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the height, width, and channels
        """
        height = test_case["height"]
        width = test_case["width"]
        channels = test_case["channels"]
        return [height, width, channels] 