import torch
import ctypes
import math
from typing import List, Dict, Tuple, Any

from problem import Problem


class bilateral_filter(Problem):
    """Bilateral filter problem for edge-preserving image smoothing."""
    
    def __init__(self):
        super().__init__(
            name="bilateral-filter"
        )
    
    def reference_solution(self, image: torch.Tensor, sigma_spatial: float, sigma_color: float, kernel_size: int) -> torch.Tensor:
        """
        PyTorch implementation of bilateral filter.
        
        Args:
            image: Input grayscale image of shape (height, width)
            sigma_spatial: Spatial standard deviation for Gaussian kernel
            sigma_color: Color standard deviation for intensity differences
            kernel_size: Size of the filter kernel (must be odd)
            
        Returns:
            Filtered image of shape (height, width)
        """
        with torch.no_grad():
            height, width = image.shape
            filtered = torch.zeros_like(image)
            
            # Create spatial Gaussian kernel
            half_kernel = kernel_size // 2
            spatial_kernel = torch.zeros((kernel_size, kernel_size), device=image.device, dtype=image.dtype)
            
            for i in range(kernel_size):
                for j in range(kernel_size):
                    di = i - half_kernel
                    dj = j - half_kernel
                    spatial_kernel[i, j] = math.exp(-(di*di + dj*dj) / (2 * sigma_spatial * sigma_spatial))
            
            # Apply bilateral filter
            for y in range(height):
                for x in range(width):
                    center_intensity = image[y, x]
                    weighted_sum = 0.0
                    weight_sum = 0.0
                    
                    for ky in range(kernel_size):
                        for kx in range(kernel_size):
                            ny = y + ky - half_kernel
                            nx = x + kx - half_kernel
                            
                            # Check bounds
                            if 0 <= ny < height and 0 <= nx < width:
                                neighbor_intensity = image[ny, nx]
                                
                                # Spatial weight
                                spatial_weight = spatial_kernel[ky, kx]
                                
                                # Color weight
                                color_diff = center_intensity - neighbor_intensity
                                color_weight = math.exp(-(color_diff * color_diff) / (2 * sigma_color * sigma_color))
                                
                                # Combined weight
                                weight = spatial_weight * color_weight
                                
                                weighted_sum += weight * neighbor_intensity
                                weight_sum += weight
                    
                    filtered[y, x] = weighted_sum / weight_sum if weight_sum > 0 else center_intensity
            
            return filtered
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for bilateral filter.
        
        Returns:
            List of test case dictionaries with varying image sizes and parameters
        """
        test_cases = [
            {
                "name": "512x512_smooth",
                "height": 512,
                "width": 512,
                "sigma_spatial": 3.0,
                "sigma_color": 50.0,
                "kernel_size": 15,
                "create_inputs": lambda h=512, w=512: self._create_noisy_image(h, w, dtype)
            },
            {
                "name": "1024x768_medium",
                "height": 1024,
                "width": 768,
                "sigma_spatial": 5.0,
                "sigma_color": 30.0,
                "kernel_size": 21,
                "create_inputs": lambda h=1024, w=768: self._create_noisy_image(h, w, dtype)
            },
            {
                "name": "1920x1080_sharp",
                "height": 1920,
                "width": 1080,
                "sigma_spatial": 2.0,
                "sigma_color": 80.0,
                "kernel_size": 11,
                "create_inputs": lambda h=1920, w=1080: self._create_noisy_image(h, w, dtype)
            },
            {
                "name": "2048x2048_large",
                "height": 2048,
                "width": 2048,
                "sigma_spatial": 4.0,
                "sigma_color": 40.0,
                "kernel_size": 17,
                "create_inputs": lambda h=2048, w=2048: self._create_noisy_image(h, w, dtype)
            }
        ]
        
        return test_cases
    
    def _create_noisy_image(self, height: int, width: int, dtype: torch.dtype) -> Tuple[torch.Tensor, float, float, int]:
        """Helper to create a noisy test image with geometric patterns."""
        # Create base image with geometric patterns
        y, x = torch.meshgrid(torch.arange(height, device="cuda", dtype=dtype), 
                             torch.arange(width, device="cuda", dtype=dtype), 
                             indexing='ij')
        
        # Create interesting patterns
        image = torch.zeros((height, width), device="cuda", dtype=dtype)
        
        # Add circles
        cx, cy = width // 2, height // 2
        radius = min(width, height) // 6
        circle_mask = ((x - cx) ** 2 + (y - cy) ** 2) < radius ** 2
        image[circle_mask] = 200.0
        
        # Add rectangles
        rect_mask = ((x > width//4) & (x < 3*width//4) & (y > height//4) & (y < 3*height//4))
        image[rect_mask] = 100.0
        
        # Add diagonal lines
        line_mask = torch.abs(x - y) < 5
        image[line_mask] = 255.0
        
        # Add noise
        noise = torch.randn_like(image) * 20.0
        image = torch.clamp(image + noise, 0.0, 255.0)
        
        return (image, 3.0, 50.0, 15)
    
    def generate_sample(self, dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
        """
        Generate a single sample test case for debugging.
        
        Returns:
            A dictionary containing a single test case
        """
        return {
            "name": "Sample (h=16, w=16)",
            "height": 16,
            "width": 16,
            "sigma_spatial": 2.0,
            "sigma_color": 30.0,
            "kernel_size": 7,
            "create_inputs": lambda: self._create_sample_image(dtype)
        }
    
    def _create_sample_image(self, dtype: torch.dtype) -> Tuple[torch.Tensor, float, float, int]:
        """Create a small sample image for debugging."""
        image = torch.tensor([
            [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160],
            [15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155, 165],
            [20, 30, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 30, 150, 160, 170],
            [25, 35, 200, 250, 250, 250, 250, 250, 250, 250, 250, 200, 35, 155, 165, 175],
            [30, 40, 200, 250, 100, 100, 100, 100, 100, 100, 250, 200, 40, 160, 170, 180],
            [35, 45, 200, 250, 100, 50, 50, 50, 50, 100, 250, 200, 45, 165, 175, 185],
            [40, 50, 200, 250, 100, 50, 10, 10, 50, 100, 250, 200, 50, 170, 180, 190],
            [45, 55, 200, 250, 100, 50, 10, 10, 50, 100, 250, 200, 55, 175, 185, 195],
            [50, 60, 200, 250, 100, 50, 50, 50, 50, 100, 250, 200, 60, 180, 190, 200],
            [55, 65, 200, 250, 100, 100, 100, 100, 100, 100, 250, 200, 65, 185, 195, 205],
            [60, 70, 200, 250, 250, 250, 250, 250, 250, 250, 250, 200, 70, 190, 200, 210],
            [65, 75, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 75, 195, 205, 215],
            [70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 80, 200, 210, 220],
            [75, 85, 95, 105, 115, 125, 135, 145, 155, 165, 175, 185, 85, 205, 215, 225],
            [80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 90, 210, 220, 230],
            [85, 95, 105, 115, 125, 135, 145, 155, 165, 175, 185, 195, 95, 215, 225, 235]
        ], device="cuda", dtype=dtype)
        
        return (image, 2.0, 30.0, 7)

    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the bilateral filter result is correct.
        
        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            
        Returns:
            Tuple of (is_correct, debug_info)
        """
        is_close = torch.allclose(actual_output, expected_output, rtol=1e-3, atol=1e-1)
        
        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            max_diff = torch.max(torch.abs(diff)).item()
            mean_diff = torch.mean(torch.abs(diff)).item()
            
            debug_info = {
                "max_difference": max_diff,
                "mean_difference": mean_diff,
                "output_range": [torch.min(actual_output).item(), torch.max(actual_output).item()],
                "expected_range": [torch.min(expected_output).item(), torch.max(expected_output).item()]
            }
        
        return is_close, debug_info
    
    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the bilateral filter solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # input_image
                ctypes.POINTER(ctypes.c_float),  # output_image
                ctypes.c_size_t,                 # height
                ctypes.c_size_t,                 # width
                ctypes.c_float,                  # sigma_spatial
                ctypes.c_float,                  # sigma_color
                ctypes.c_int                     # kernel_size
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
        # - kernel_size^2 iterations through the kernel
        # - Each iteration: 2 exponential calculations (spatial + color weights)
        # - 1 multiplication for combined weight
        # - 1 multiplication for weighted sum
        # - 1 addition for weighted sum
        # - 1 addition for weight sum
        # - 1 division at the end
        # Total per kernel pixel: ~6 operations (simplified estimate)
        # Total per output pixel: kernel_size^2 * 6 + 1 (final division)
        return height * width * (kernel_size * kernel_size * 6 + 1)
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the height, width, sigma_spatial, sigma_color, and kernel_size
        """
        return [
            test_case["height"],
            test_case["width"],
            test_case["sigma_spatial"],
            test_case["sigma_color"],
            test_case["kernel_size"]
        ] 