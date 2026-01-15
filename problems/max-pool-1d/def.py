import torch
import ctypes
from typing import List, Dict, Tuple, Any
from problem import Problem
from tolerances import tol_for

class max_pool_1d(Problem):
    """1D max pooling problem."""

    numeric_category = "LOCAL"
    
    def __init__(self):
        super().__init__(
            name="max-pool-1d"
        )
    
    def reference_solution(self, input_tensor: torch.Tensor, kernel_size: int, 
                         stride: int, padding: int, dilation: int) -> torch.Tensor:
        """
        PyTorch implementation of 1D max pooling.
        
        Args:
            input_tensor: Input tensor of shape (H)
            kernel_size: Size of the pooling window
            stride: Stride of the pooling window
            padding: Padding to be applied before pooling
            dilation: Spacing between kernel elements (controls the gap between elements in the kernel)
            
        Returns:
            Result of max pooling
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            input_reshaped = input_tensor.view(1, 1, input_tensor.size(0))
            
            result = torch.nn.functional.max_pool1d(
                input_reshaped,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            )
            
            return result.view(result.size(2))
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for 1D max pooling.
        
        Returns:
            List of test case dictionaries with varying sizes
        """
        test_configs = [
            (2**21, 7, 4, 3, 1),  # H=2^21, k=7, S=4, P=3, d=1
            (2**22, 2, 1, 0, 1),  # H=2^22, k=2, S=1, P=0, d=1
            (2**23, 3, 2, 1, 1),  # H=2^23, k=3, S=2, P=1, d=1
            (2**24, 4, 2, 1, 2),  # H=2^24, k=4, S=2, P=1, d=2
            (2**25, 3, 1, 1, 1),  # H=2^25, k=3, S=1, P=1, d=1
            (2**26, 5, 3, 2, 1),  # H=2^26, k=5, S=3, P=2, d=1
        ]
        return [
            {
                "name": f"H={h}, K={k}, S={s}, P={p}, d={d}",
                "size": h,
                "kernel_size": k,
                "stride": s,
                "padding": p,
                "dilation": d,
                "create_inputs": lambda size=h, kernel_size=k, stride=s, padding=p, dilation=d: (
                    torch.rand((size), device="cuda", dtype=dtype) * 10.0 - 5.0,  # uniform [-5, 5]
                    kernel_size, 
                    stride, 
                    padding,
                    dilation
                )
            }
            for h, k, s, p, d in test_configs
        ]
    
    def generate_sample(self, dtype: torch.dtype = torch.float32) -> List[Dict[str, Any]]:
        """
        Generate sample test cases for 1D max pooling with predictable inputs.

        Returns:
            List of sample test case dictionaries.
        """

        return {
            "name": "sample_basic_3x3",
            "size": 16,
            "kernel_size": 3,
            "stride": 1,
            "padding": 0,
            "dilation": 1,
            "create_inputs": lambda size_val=16, k_val=3, s_val=1, p_val=0, d_val=1, dtype_val=dtype: (
                torch.rand((size_val), device="cuda", dtype=dtype_val) * 10.0 - 5.0,
                k_val,
                s_val,
                p_val,
                d_val
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
        tol = tol_for(dtype, self.numeric_category)
        if tol is None:
            is_close = torch.equal(actual_output, expected_output)
        else:
            is_close = torch.allclose(actual_output, expected_output, rtol=tol.rtol, atol=tol.atol)
        
        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            max_diff = torch.max(torch.abs(diff)).item()
            mean_diff = torch.mean(torch.abs(diff)).item()
            
            # Find indices of largest differences
            flat_diff = diff.flatten()
            _, top_indices = torch.topk(torch.abs(flat_diff), min(5, flat_diff.numel()))
            
            # Convert flat indices back to 1D coordinates
            sample_diffs = {}
            for i, idx in enumerate(top_indices):
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
        Get the function signature for the 1D max pooling solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # input_tensor
                ctypes.c_size_t,                 # kernel_size
                ctypes.c_size_t,                 # stride
                ctypes.c_size_t,                 # padding
                ctypes.c_size_t,                 # dilation
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_size_t,                 # size (H)
            ],
            "restype": None
        }
    
    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the number of floating point operations for the problem.
        
        Args:
            test_case: The test case dictionary
            
        IMPORTANT: For max pooling, we count comparisons as FLOPs.
        Each output element requires (kernel_size - 1) comparisons.
        
        Returns:
            Number of floating point operations
        """
        H = test_case["size"]
        K = test_case["kernel_size"]
        S = test_case["stride"]
        P = test_case["padding"]
        D = test_case["dilation"]
        
        # Calculate output dimensions
        H_out = ((H + 2 * P - D * (K - 1) - 1) // S) + 1
        # Each output element requires K-1 comparisons
        comparisons_per_output = K - 1
        
        # Total FLOPs (comparisons) for the entire output
        return H_out * comparisons_per_output
    
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
