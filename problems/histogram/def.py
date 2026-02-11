import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class histogram(Problem):
    """Histogram computation problem."""
    
    is_exact = True
    
    def __init__(self):
        super().__init__(
            name="histogram"
        )
    
    def reference_solution(self, input_image: torch.Tensor, num_bins: int) -> torch.Tensor:
        """
        PyTorch implementation of histogram computation.
        
        Args:
            input_image: Input grayscale image of shape (height, width)
            num_bins: Number of histogram bins (typically 256 for 8-bit images)
            
        Returns:
            Histogram array of shape (num_bins,) containing pixel counts
        """
        with torch.no_grad():
            clamped_input = torch.clamp(input_image, 0, num_bins - 1)
            indices = clamped_input.long().flatten()
            histogram = torch.bincount(indices, minlength=num_bins).float()
            return histogram
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for histogram computation.
        
        Returns:
            List of test case dictionaries with varying image sizes and bin counts
        """
        image_sizes = [
            (2560, 1440),
            (2048, 2048),
            (4096, 4096)
        ]
        
        bin_counts = [64, 128, 256]
        
        test_cases = []
        for height, width in image_sizes:
            for num_bins in bin_counts:
                seed = Problem.get_seed(f"{self.name}_H={height}_W={width}_bins={num_bins}")
                test_cases.append({
                    "name": f"{height}x{width}, bins={num_bins}",
                    "height": height,
                    "width": width,
                    "num_bins": num_bins,
                    "create_inputs": lambda h=height, w=width, b=num_bins, seed=seed, dtype=dtype: (
                        torch.randint(
                            0, b, (h, w),
                            device="cuda",
                            dtype=dtype,
                            generator=torch.Generator(device="cuda").manual_seed(seed),
                        ),
                        b
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
        num_bins = 256
        return {
            "name": "Sample (h=8, w=8, bins=256)",
            "height": image_size[0],
            "width": image_size[1],
            "num_bins": num_bins,
            "create_inputs": lambda h=image_size[0], w=image_size[1], b=num_bins: (
                torch.randint(0, b, (h, w), device="cuda", dtype=dtype),
                b
            )
        }
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the histogram result is correct.
        
        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            
        Returns:
            Tuple of (is_correct, debug_info)
        """
        # For histograms, we expect exact matches
        is_equal = torch.all(torch.eq(actual_output, expected_output))
        
        debug_info = {}
        if not is_equal:
            # Find bins with mismatched counts
            diff = actual_output - expected_output
            nonzero_diff = torch.nonzero(diff != 0)
            
            sample_diffs = {}
            for i in range(min(5, nonzero_diff.size(0))):
                bin_idx = nonzero_diff[i, 0].item()
                sample_diffs[f"bin_{bin_idx}"] = {
                    "expected": expected_output[bin_idx].item(),
                    "actual": actual_output[bin_idx].item(),
                    "diff": diff[bin_idx].item()
                }
                        
            debug_info = {
                "total_expected": expected_output.sum().item(),
                "total_actual": actual_output.sum().item(),
                "mismatched_bins": nonzero_diff.size(0),
                "sample_differences": sample_diffs
            }
        
        return is_equal, debug_info
    
    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the histogram solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # image
                ctypes.c_int,                    # num_bins
                ctypes.POINTER(ctypes.c_float),  # histogram
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
        
        # For each pixel:
        # - 1 memory access to read pixel value
        # - 1 clamp operation (2 comparisons)
        # - 1 atomic increment operation
        # Total: 4 operations per pixel
        return height * width * 4
    
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
        num_bins = test_case["num_bins"]
        
        # Naive histogram:
        # 1. Read input image → height*width
        # 2. Read histogram bins (for each pixel, read current bin value) → height*width (reads to bins)
        # 3. Write histogram bins (increment) → height*width (writes to bins)
        # 4. Read histogram bins (final read) → num_bins
        # 5. Write output histogram → num_bins
        
        dtype_bytes = 4  # 4 bytes per float32 element
        return (height * width +        # read input
                height * width +        # read bins during increment
                height * width +        # write bins during increment
                num_bins +              # read final bins
                num_bins) * dtype_bytes  # write output
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the height, width, and number of bins
        """
        height = test_case["height"]
        width = test_case["width"]
        return [height, width] 