import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class threshold(Problem):
    """Image thresholding problem."""
    
    def __init__(self):
        super().__init__(
            name="threshold"
        )
    
    def reference_solution(self, input_image: torch.Tensor, threshold_value: float) -> torch.Tensor:
        """
        PyTorch implementation of binary thresholding.
        
        Args:
            input_image: Input grayscale image of shape (height, width)
            threshold_value: Threshold value for binarization
            
        Returns:
            Binary image of shape (height, width) with values 0 or 255
        """
        with torch.no_grad():
            # Apply binary thresholding
            return torch.where(input_image > threshold_value, 
                              torch.tensor(255.0, device=input_image.device, dtype=input_image.dtype),
                              torch.tensor(0.0, device=input_image.device, dtype=input_image.dtype))
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for image thresholding.
        
        Returns:
            List of test case dictionaries with varying image sizes and threshold values
        """
        image_sizes = [
            (1024, 768),    
            (1920, 1080),   
            (3840, 2160)    
        ]
        
        threshold_values = [64.0, 128.0, 192.0]
        
        test_cases = []
        for height, width in image_sizes:
            for threshold in threshold_values:
                seed = Problem.get_seed(f"{self.name}_H={height}_W={width}_t={threshold}")
                test_cases.append({
                    "name": f"{height}x{width}, threshold={threshold}",
                    "height": height,
                    "width": width,
                    "threshold_value": threshold,
                    "create_inputs": lambda h=height, w=width, t=threshold, seed=seed, dtype=dtype: (
                        *(lambda g: (
                            # Create a random grayscale image with values in [0, 255]
                            torch.rand((h, w), device="cuda", dtype=dtype, generator=g) * 255.0,
                        ))(torch.Generator(device="cuda").manual_seed(seed)),
                        t
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
        threshold_value = 128.0
        return {
            "name": "Sample (h=8, w=8, threshold=128.0)",
            "height": image_size[0],
            "width": image_size[1],
            "create_inputs": lambda h=image_size[0], w=image_size[1], t=threshold_value: (
                torch.rand((h, w), device="cuda", dtype=dtype) * 255.0,
                t
            )
        }
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the thresholding result is correct.
        
        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            
        Returns:
            Tuple of (is_correct, debug_info)
        """
        # For binary thresholding, we expect exact matches
        is_equal = torch.all(torch.eq(actual_output, expected_output))
        
        debug_info = {}
        if not is_equal:
            # Count mismatched pixels
            mismatched = (actual_output != expected_output).sum().item()
            total_pixels = actual_output.numel()
            
            # Find indices of some mismatches for debugging
            mismatches = torch.nonzero(actual_output != expected_output)
            sample_diffs = {}
            
            for i in range(min(5, mismatches.size(0))):
                row, col = mismatches[i].tolist()
                sample_diffs[f"({row}, {col})"] = {
                    "expected": expected_output[row, col].item(),
                    "actual": actual_output[row, col].item()
                }
                        
            debug_info = {
                "mismatched_pixels": mismatched,
                "total_pixels": total_pixels,
                "mismatch_percentage": (mismatched / total_pixels) * 100,
                "sample_mismatches": sample_diffs
            }
        
        return is_equal, debug_info
    
    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the image thresholding solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # image
                ctypes.c_float,                  # threshold_value
                ctypes.POINTER(ctypes.c_float),  # thresholded_image
                ctypes.c_size_t,                 # height
                ctypes.c_size_t,                 # width
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
        # - 1 comparison operation
        # Total: 1 FLOP per pixel
        return height * width
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the height, width, and threshold value
        """
        height = test_case["height"]
        width = test_case["width"]
        return [height, width] 