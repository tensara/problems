import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem

class avg_pool_1d(Problem):
    """1D average pooling problem."""
    
    is_exact = False
    
    def __init__(self):
        super().__init__(
            name="avg-pool-1d"
        )
    
    def reference_solution(self, input_tensor: torch.Tensor, kernel_size: int, 
                         stride: int, padding: int) -> torch.Tensor:
        """
        PyTorch implementation of 1D average pooling.
        
        Args:
            input_tensor: Input tensor of shape (H)
            kernel_size: Size of the pooling window
            stride: Stride of the pooling window
            padding: Padding to be applied before pooling
            
        Returns:
            Result of average pooling
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=input_tensor.dtype):
            input_reshaped = input_tensor.view(1, 1, input_tensor.size(0))
            
            result = torch.nn.functional.avg_pool1d(
                input_reshaped,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            
            return result.view(result.size(2))
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for 1D average pooling.
        
        Returns:
            List of test case dictionaries with varying sizes
        """
        test_configs = [
            (2**21, 7, 4, 3),  # H=2^21, k=7, S=4, P=3
            (2**22, 2, 1, 0),  # H=2^22, k=2, S=1, P=0
            (2**23, 3, 2, 1),  # H=2^23, k=3, S=2, P=1
            (2**24, 4, 2, 1),  # H=2^24, k=4, S=2, P=1
            (2**25, 3, 1, 1),  # H=2^25, k=3, S=1, P=1
            (2**26, 5, 3, 2),  # H=2^26, k=5, S=3, P=2
        ]

        test_cases = []
        for h, k, s, p in test_configs:
            seed = Problem.get_seed(f"{self.name}_H={h}_K={k}_S={s}_P={p}")
            test_cases.append({
                "name": f"H={h}, K={k}, S={s}, P={p}",
                "size": h,
                "kernel_size": k,
                "stride": s,
                "padding": p,
                "create_inputs": lambda size=h, kernel_size=k, stride=s, padding=p, seed=seed, dtype=dtype: (
                    *(lambda g: (
                        torch.rand((size), device="cuda", dtype=dtype, generator=g) * 2.0 - 1.0, # uniform [-1, 1]
                    ))(torch.Generator(device="cuda").manual_seed(seed)),
                    kernel_size, 
                    stride, 
                    padding
                )
            })
        return test_cases
    
    def generate_sample(self, dtype: torch.dtype = torch.float32) -> List[Dict[str, Any]]:
        """
        Generate a single sample test case for debugging or interactive runs.
        
        Returns:
            A list containing a single test case dictionary
        """
        h, k, s, p = (16, 3, 2, 1) # Sample configuration
        return {
            "name": f"H={h}, K={k}, S={s}, P={p}",
            "size": h,
            "kernel_size": k,
            "stride": s,
            "padding": p,
            "create_inputs": lambda size=h, kernel_size=k, stride=s, padding=p: (
                torch.arange(1, size + 1, device="cuda", dtype=dtype).float(), # Sequential input for easy verification
                kernel_size, 
                stride, 
                padding
            )
        }
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the average pooling result is correct.
        
        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            
        Returns:
            Tuple of (is_correct, debug_info)
        """
        is_close = torch.allclose(actual_output, expected_output, rtol=5e-3, atol=2e-4)
        
        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            max_diff = torch.max(torch.abs(diff)).item()
            mean_diff = torch.mean(torch.abs(diff)).item()
            
            # Find indices of largest differences
            flat_diff = diff.flatten()
            _, top_indices = torch.topk(torch.abs(flat_diff), min(5, flat_diff.numel()))
            
            sample_diffs = {}
            for _, idx in enumerate(top_indices):
                sample_diffs[f"({idx})"] = {
                    "expected": expected_output[idx].item(),
                    "actual": actual_output[idx].item(),
                    "diff": diff[idx].item()
                }
            
            debug_info = {
                "max_difference": max_diff,
                "mean_difference": mean_diff,
                "sample_differences": sample_diffs
            }
        
        return is_close, debug_info
    
    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the 2D average pooling solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # input_tensor
                ctypes.c_size_t,                 # kernel_size
                ctypes.c_size_t,                 # stride
                ctypes.c_size_t,                 # padding
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_size_t,                 # height (H)
            ],
            "restype": None
        }
    
    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the number of floating point operations for the problem.
        
        Args:
            test_case: The test case dictionary
            
        For average pooling, we count:
        - One addition per element in the kernel window
        - One division per output element
        
        Returns:
            Number of floating point operations
        """
        H = test_case["size"]
        K = test_case["kernel_size"]
        S = test_case["stride"]
        P = test_case["padding"]
        
        # Calculate output dimensions
        H_out = ((H + 2 * P - K) // S) + 1
        
        # Each output element requires:
        # - (K - 1) additions for summing the window
        # - 1 division for computing the average
        ops_per_output = K - 1 + 1
        
        # Total FLOPs for the entire output
        return H_out * ops_per_output
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the image size H
        """
        return [
            test_case["size"],
        ]
