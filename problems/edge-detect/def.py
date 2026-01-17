import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class edge_detect(Problem):
    """Edge detection problem."""
    
    def __init__(self):
        super().__init__(
            name="edge-detect"
        )
    
    def reference_solution(self, input_image: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of simple edge detection using gradients.
        
        Args:
            input_image: Input grayscale image of shape (height, width)
            
        Returns:
            Edge detected image of shape (height, width)
        """
        with torch.no_grad():
            h, w = input_image.shape
            
            gx_kernel = torch.tensor([[-1, 0, 1]], device=input_image.device) / 2.0
            gy_kernel = torch.tensor([[-1], [0], [1]], device=input_image.device) / 2.0
            
            input_reshaped = input_image.view(1, 1, h, w)
            gx_kernel = gx_kernel.view(1, 1, 1, 3)
            gy_kernel = gy_kernel.view(1, 1, 3, 1)
            
            gx = torch.nn.functional.conv2d(input_reshaped, gx_kernel, padding=(0, 1))
            gy = torch.nn.functional.conv2d(input_reshaped, gy_kernel, padding=(1, 0))
            
            magnitude = torch.sqrt(gx**2 + gy**2).squeeze()
            
            magnitude[0, :] = 0
            magnitude[-1, :] = 0
            magnitude[:, 0] = 0
            magnitude[:, -1] = 0
            
            if magnitude.max() > 0:
                magnitude = (magnitude / magnitude.max()) * 255.0
            
            return magnitude
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for edge detection.
        
        Returns:
            List of test case dictionaries with varying image sizes
        """
        image_sizes = [
            (1024, 768),    
            (1920, 1080),
            (2048, 2048),
            (2560, 1440),
            (4096, 4096)
        ]
        
        test_cases = []
        for height, width in image_sizes:
            seed = Problem.get_seed(f"{self.name}_H={height}_W={width}")
            test_cases.append({
                "name": f"{height}x{width}",
                "height": height,
                "width": width,
                "create_inputs": lambda h=height, w=width, seed=seed, dtype=dtype: (
                    (lambda g: (
                        # Create a random grayscale image with values in [0, 255]
                        torch.rand((h, w), device="cuda", dtype=dtype, generator=g) * 255.0,
                    ))(torch.Generator(device="cuda").manual_seed(seed))
                )
            })
        
        return test_cases
    
    def generate_sample(self, dtype: torch.dtype = torch.float32) -> List[Dict[str, Any]]:
        """
        Generate a single sample test case for debugging or interactive runs.
        
        Returns:
            A list containing a single test case dictionary
        """
        image_size = (16, 16)
        return {
            "name": "Sample (h=16, w=16)",
            "height": image_size[0],
            "width": image_size[1],
            "create_inputs": lambda h=image_size[0], w=image_size[1]: (
                torch.rand((h, w), device="cuda", dtype=dtype) * 255.0,
            )
        }
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the edge detection result is correct.
        
        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            
        Returns:
            Tuple of (is_correct, debug_info)
        """
        is_close = torch.allclose(actual_output, expected_output, rtol=1e-3, atol=1e-3)
        
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
        Get the function signature for the edge detection solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # input_image
                ctypes.POINTER(ctypes.c_float),  # output_image
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
        
        # For each interior pixel (excluding 1-pixel border):
        # - 2 subtractions for Gx (right - left)
        # - 2 subtractions for Gy (bottom - top)  
        # - 1 division for Gx/2, 1 division for Gy/2
        # - 2 multiplications for Gx^2 and Gy^2
        # - 1 addition for Gx^2 + Gy^2
        # - 1 square root for magnitude
        # Total: 10 FLOPs per interior pixel
        interior_pixels = (height - 2) * (width - 2)
        return interior_pixels * 10
    
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
        
        # Input image: height*width elements
        # Output image: height*width elements (same size)
        dtype_bytes = 4  # 4 bytes per float32 element
        return (height * width + height * width) * dtype_bytes
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the height and width
        """
        height = test_case["height"]
        width = test_case["width"]
        return [height, width] 