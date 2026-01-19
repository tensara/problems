import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem

class conv_square_3d(Problem):
    """3D convolution problem with square input and square kernel."""
    
    is_exact = False
    
    def __init__(self):
        super().__init__(
            name="conv-square-3d"
        )
    
    def reference_solution(self, input_volume: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of 3D convolution with square input and square kernel.
        
        Args:
            input_volume: Input volume tensor of shape (D, H, W)
            kernel: Convolution kernel tensor of shape (K, K, K) where K is the kernel size
            
        Returns:
            Result of convolution with zero padding
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=input_volume.dtype):
            assert kernel.size(0) == kernel.size(1) == kernel.size(2), "Kernel must be cubic (equal dimensions)"
            
            input_reshaped = input_volume.view(1, 1, input_volume.size(0), input_volume.size(1), input_volume.size(2))
            kernel_reshaped = kernel.view(1, 1, kernel.size(0), kernel.size(1), kernel.size(2))
            
            padding = kernel.size(0) // 2
            
            result = torch.nn.functional.conv3d(
                input_reshaped, 
                kernel_reshaped, 
                padding=padding
            )
            
            return result.view(input_volume.size(0), input_volume.size(1), input_volume.size(2))
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for 3D convolution.
        
        Returns:
            List of test case dictionaries with varying sizes
        """
        test_configs = [
            (64, 64, 64, 3),    # Small volume, small kernel
            (32, 32, 32, 9),     # Small volume, large kernel
            (96, 96, 96, 11),    # Medium volume, larger kernel
            (128, 128, 128, 5),  # Medium volume, medium kernel
            (256, 256, 256, 7),  # Large volume, large kernel
            (512, 512, 512, 9),  # Large volume, large kernel
        ]
        
        test_cases = []
        for size, _, _, k in test_configs:
            seed = Problem.get_seed(f"{self.name}_size={size}_K={k}")
            test_cases.append({
                "name": f"D=H=W={size}, K={k}",
                "size": size,
                "kernel_size": k,
                "create_inputs": lambda size=size, k=k, seed=seed, dtype=dtype: (
                    *(lambda g: (
                        torch.rand((size, size, size), device="cuda", dtype=dtype, generator=g) * 2.0 - 1.0,  # uniform [-1, 1]
                        torch.rand((k, k, k), device="cuda", dtype=dtype, generator=g) * 2.0 - 1.0  # uniform [-1, 1]
                    ))(torch.Generator(device="cuda").manual_seed(seed)),
                )
            })
        return test_cases
    
    def generate_sample(self, dtype: torch.dtype = torch.float32) -> List[Dict[str, Any]]:
        """
        Generate a single sample test case for debugging or interactive runs.
        
        Returns:
            A list containing a single test case dictionary
        """
        size, k = (4, 3) # Sample configuration (kernel size must be odd)
        return {
            "name": f"D=H=W={size}, K={k}",
            "size": size,
            "kernel_size": k,
            "create_inputs": lambda size=size, k=k: (
                torch.arange(1, size**3 + 1, device="cuda", dtype=dtype).float().view(size, size, size), # Sequential input
                torch.ones((k, k, k), device="cuda", dtype=dtype) # Simple kernel
            )
        }
    
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
            
            # Convert flat indices back to 3D coordinates
            d, h, w = expected_output.shape
            sample_diffs = {}
            for i, idx in enumerate(top_indices):
                idx = idx.item()
                depth = idx // (h * w)
                height = (idx % (h * w)) // w
                width = idx % w
                sample_diffs[f"({depth}, {height}, {width})"] = {
                    "expected": expected_output[depth, height, width].item(),
                    "actual": actual_output[depth, height, width].item(),
                    "diff": diff[depth, height, width].item()
                }
            
            debug_info = {
                "max_difference": max_diff,
                "mean_difference": mean_diff,
                "sample_differences": sample_diffs
            }
        
        return is_close, debug_info
    
    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the 3D convolution solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # input_volume
                ctypes.POINTER(ctypes.c_float),  # kernel
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_size_t,                 # size (D=H=W)
                ctypes.c_size_t,                 # kernel_size (K)
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
        # For 3D convolution with cubic kernel, each output voxel requires:
        # K*K*K multiplications and K*K*K-1 additions where K is the kernel size
        size = test_case["size"]  # D=H=W
        K = test_case["kernel_size"]
        
        # Total FLOPs for the entire volume: size^3 output voxels, each requiring:
        # K^3 multiplications + (K^3-1) additions = 2*K^3 - 1 FLOPs
        # Following similar convention as 2D case, we use 2*size^3*K^3
        return 2 * size * size * size * K * K * K
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the volume size (D=H=W) and kernel size K
        """
        size = test_case["size"]
        K = test_case["kernel_size"]
        return [size, K]
