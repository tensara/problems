import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class max_pool_2d(Problem):
    """2D max pooling problem."""
    
    def __init__(self):
        super().__init__(
            name="max-pool-2d"
        )
    
    def reference_solution(self, input_tensor: torch.Tensor, kernel_size: int, 
                         stride: int, padding: int, dilation: int) -> torch.Tensor:
        """
        PyTorch implementation of 2D max pooling.
        
        Args:
            input_tensor: Input tensor of shape (H, W)
            kernel_size: Size of the pooling window
            stride: Stride of the pooling window
            padding: Padding to be applied before pooling
            dilation: Spacing between kernel elements
            
        Returns:
            Result of max pooling
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            input_reshaped = input_tensor.view(1, 1, input_tensor.size(0), input_tensor.size(1))
            
            result = torch.nn.functional.max_pool2d(
                input_reshaped,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            )
            
            return result.view(result.size(2), result.size(3))
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for 2D max pooling.
        
        Returns:
            List of test case dictionaries with varying sizes
        """
        test_configs = [
            (4096, 4096, 2, 2, 0, 1),
            (8192, 8192, 3, 2, 1, 1),
            (16384, 16384, 4, 4, 2, 1),
            (1024, 1024, 3, 3, 1, 2),
            (2048, 2048, 5, 2, 2, 1),
            (4096, 4096, 7, 3, 3, 1)
        ]
        
        test_cases = []
        for h, w, k, s, p, d in test_configs:
            seed = Problem.get_seed(f"{self.name}_H={h}_W={w}_K={k}_S={s}_P={p}_D={d}")
            test_cases.append({
                "name": f"H={h}, W={w}, K={k}, S={s}, P={p}, D={d}",
                "height": h,
                "width": w,
                "kernel_size": k,
                "stride": s,
                "padding": p,
                "dilation": d,
                "create_inputs": lambda h=h, w=w, k=k, s=s, p=p, d=d, seed=seed, dtype=dtype: (
                    *(lambda g: (
                        torch.rand((h, w), device="cuda", dtype=dtype, generator=g) * 10.0 - 5.0,  # uniform [-5, 5]
                    ))(torch.Generator(device="cuda").manual_seed(seed)),
                    k, s, p, d
                )
            })
        return test_cases
    
    def generate_sample(self, dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
        """
        Generate a single sample test case for 2D max pooling with predictable input.

        Returns:
            A dictionary containing the sample test case
        """
        return {
            "name": "sample_basic_3x3",
            "height": 8,
            "width": 8, 
            "kernel_size": 3,
            "stride": 1,
            "padding": 0,
            "dilation": 1,
            "create_inputs": lambda h=8, w=8, k=3, s=1, p=0, d=1: (
                torch.rand((h, w), device="cuda", dtype=dtype) * 10.0 - 5.0,
                k,
                s,
                p,
                d
            )
        }
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the max pooling result is correct.
        
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
        Get the function signature for the 2D max pooling solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # input_tensor
                ctypes.c_size_t,                 # kernel_size
                ctypes.c_size_t,                 # stride
                ctypes.c_size_t,                 # padding
                ctypes.c_size_t,                  # dilation
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_size_t,                 # height (H)
                ctypes.c_size_t,                 # width (W)

            ],
            "restype": None
        }
    
    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the number of floating point operations for the problem.
        
        Args:
            test_case: The test case dictionary
            
        IMPORTANT: For max pooling, we count comparisons as FLOPs.
        Each output element requires (kernel_size * kernel_size - 1) comparisons.
        
        Returns:
            Number of floating point operations
        """
        H = test_case["height"]
        W = test_case["width"]
        K = test_case["kernel_size"]
        S = test_case["stride"]
        P = test_case["padding"]
        
        # Calculate output dimensions
        H_out = ((H + 2 * P - K) // S) + 1
        W_out = ((W + 2 * P - K) // S) + 1
        
        # Each output element requires K*K-1 comparisons
        comparisons_per_output = K * K - 1
        
        # Total FLOPs (comparisons) for the entire output
        return H_out * W_out * comparisons_per_output
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the image height H, width W, kernel_size, stride, padding, and dilation
        """
        return [
            test_case["height"],
            test_case["width"],
        ]
