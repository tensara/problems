import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class mandelbrot_set(Problem):
    """Mandelbrot set computation problem."""
    
    def __init__(self):
        super().__init__(
            name="mandelbrot-set"
        )
    
    def reference_solution(self, real_min: float, real_max: float, imag_min: float, imag_max: float, width: int, height: int, max_iter: int) -> torch.Tensor:
        """
        PyTorch implementation of Mandelbrot set computation.
        
        Args:
            real_min: Minimum real value of complex plane
            real_max: Maximum real value of complex plane
            imag_min: Minimum imaginary value of complex plane
            imag_max: Maximum imaginary value of complex plane
            width: Width of output image
            height: Height of output image
            max_iter: Maximum number of iterations
            
        Returns:
            Iteration counts for each pixel (height x width)
        """
        with torch.no_grad():
            # Create coordinate grids
            real_vals = torch.linspace(real_min, real_max, width, device="cuda", dtype=torch.float32)
            imag_vals = torch.linspace(imag_min, imag_max, height, device="cuda", dtype=torch.float32)
            
            # Create meshgrid for complex plane
            real_grid, imag_grid = torch.meshgrid(real_vals, imag_vals, indexing='xy')
            real_grid = real_grid.T  # Transpose to get (height, width)
            imag_grid = imag_grid.T
            
            # Initialize result
            result = torch.zeros((height, width), device="cuda", dtype=torch.float32)
            
            # Compute Mandelbrot set
            for y in range(height):
                for x in range(width):
                    c_real = real_grid[y, x]
                    c_imag = imag_grid[y, x]
                    
                    # Initialize z = 0
                    z_real = 0.0
                    z_imag = 0.0
                    
                    # Iterate z = z^2 + c
                    for i in range(max_iter):
                        # Check if |z| > 2 (diverged)
                        if z_real * z_real + z_imag * z_imag > 4.0:
                            result[y, x] = i
                            break
                        
                        # Compute z = z^2 + c
                        z_real_new = z_real * z_real - z_imag * z_imag + c_real
                        z_imag_new = 2.0 * z_real * z_imag + c_imag
                        z_real = z_real_new
                        z_imag = z_imag_new
                    else:
                        # Didn't diverge within max_iter iterations
                        result[y, x] = max_iter
            
            return result
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for Mandelbrot set computation.
        
        Returns:
            List of test case dictionaries with varying image sizes and regions
        """
        test_cases = [
            {
                "name": "classic_1024x1024",
                "real_min": -2.5,
                "real_max": 1.5,
                "imag_min": -2.0,
                "imag_max": 2.0,
                "width": 1024,
                "height": 1024,
                "max_iter": 1000,
                "create_inputs": lambda: self._create_mandelbrot_inputs(-2.5, 1.5, -2.0, 2.0, 1024, 1024, 1000)
            },
            {
                "name": "high_res_2048x2048",
                "real_min": -2.5,
                "real_max": 1.5,
                "imag_min": -2.0,
                "imag_max": 2.0,
                "width": 2048,
                "height": 2048,
                "max_iter": 1000,
                "create_inputs": lambda: self._create_mandelbrot_inputs(-2.5, 1.5, -2.0, 2.0, 2048, 2048, 1000)
            },
            {
                "name": "zoom_seahorse_1024x1024",
                "real_min": -0.75,
                "real_max": -0.73,
                "imag_min": 0.095,
                "imag_max": 0.115,
                "width": 1024,
                "height": 1024,
                "max_iter": 2000,
                "create_inputs": lambda: self._create_mandelbrot_inputs(-0.75, -0.73, 0.095, 0.115, 1024, 1024, 2000)
            },
            {
                "name": "ultra_wide_3840x2160",
                "real_min": -2.5,
                "real_max": 1.5,
                "imag_min": -1.125,
                "imag_max": 1.125,
                "width": 3840,
                "height": 2160,
                "max_iter": 1000,
                "create_inputs": lambda: self._create_mandelbrot_inputs(-2.5, 1.5, -1.125, 1.125, 3840, 2160, 1000)
            }
        ]
        
        return test_cases
    
    def _create_mandelbrot_inputs(self, real_min: float, real_max: float, imag_min: float, imag_max: float, width: int, height: int, max_iter: int) -> Tuple[float, float, float, float, int, int, int]:
        """Helper to create inputs for Mandelbrot set computation."""
        return (real_min, real_max, imag_min, imag_max, width, height, max_iter)
    
    def generate_sample(self, dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
        """
        Generate a single sample test case for debugging.
        
        Returns:
            A dictionary containing a single test case
        """
        return {
            "name": "Sample (16x16)",
            "real_min": -2.0,
            "real_max": 2.0,
            "imag_min": -2.0,
            "imag_max": 2.0,
            "width": 16,
            "height": 16,
            "max_iter": 100,
            "create_inputs": lambda: self._create_mandelbrot_inputs(-2.0, 2.0, -2.0, 2.0, 16, 16, 100)
        }

    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the Mandelbrot set computation result is correct.
        
        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            
        Returns:
            Tuple of (is_correct, debug_info)
        """
        # Mandelbrot computation should be exact (integer iteration counts)
        is_exact = torch.equal(actual_output, expected_output)
        
        debug_info = {}
        if not is_exact:
            diff = actual_output - expected_output
            diff_mask = (diff != 0.0)
            num_different = torch.sum(diff_mask).item()
            
            if num_different > 0:
                max_diff = torch.max(torch.abs(diff)).item()
                mean_diff = torch.mean(torch.abs(diff[diff_mask])).item()
                
                # Find some sample differences
                diff_indices = torch.nonzero(diff_mask)[:5]  # First 5 differences
                sample_diffs = {}
                for i, (y, x) in enumerate(diff_indices):
                    y_idx, x_idx = y.item(), x.item()
                    sample_diffs[f"({y_idx}, {x_idx})"] = {
                        "expected": expected_output[y_idx, x_idx].item(),
                        "actual": actual_output[y_idx, x_idx].item(),
                        "diff": diff[y_idx, x_idx].item()
                    }
                
                debug_info = {
                    "num_different_pixels": num_different,
                    "total_pixels": expected_output.numel(),
                    "max_difference": max_diff,
                    "mean_difference": mean_diff,
                    "sample_differences": sample_diffs
                }
        
        return is_exact, debug_info
    
    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the Mandelbrot set solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # output_image
                ctypes.c_float,                  # real_min
                ctypes.c_float,                  # real_max
                ctypes.c_float,                  # imag_min
                ctypes.c_float,                  # imag_max
                ctypes.c_int,                    # width
                ctypes.c_int,                    # height
                ctypes.c_int                     # max_iter
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
        width = test_case["width"]
        height = test_case["height"]
        max_iter = test_case["max_iter"]
        
        # For each pixel:
        # - Calculate complex coordinates: 2 operations
        # - For each iteration (worst case max_iter):
        #   - Check divergence: 3 operations (2 muls + 1 add for |z|^2, 1 comparison)
        #   - Compute z^2 + c: 6 operations (z_real^2, z_imag^2, 2*z_real*z_imag, sub, 2 adds)
        # Total per pixel (worst case): 2 + max_iter * 9
        return width * height * (2 + max_iter * 9)
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the Mandelbrot set parameters
        """
        return [
            test_case["real_min"],
            test_case["real_max"], 
            test_case["imag_min"],
            test_case["imag_max"],
            test_case["width"],
            test_case["height"],
            test_case["max_iter"]
        ] 