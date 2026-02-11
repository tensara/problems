import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem

class avg_pool_3d(Problem):
    """3D average pooling problem."""
    
    is_exact = False
    
    def __init__(self):
        super().__init__(
            name="avg-pool-3d"
        )
    
    def reference_solution(self, input_tensor: torch.Tensor, kernel_size: int, 
                         stride: int, padding: int) -> torch.Tensor:
        """
        PyTorch implementation of 3D average pooling.
        
        Args:
            input_tensor: Input tensor of shape (H, W, D)
            kernel_size: Size of the pooling window
            stride: Stride of the pooling window
            padding: Padding to be applied before pooling
            
        Returns:
            Result of average pooling
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=input_tensor.dtype):
            input_reshaped = input_tensor.view(1, 1, input_tensor.size(0), input_tensor.size(1), input_tensor.size(2))
            
            result = torch.nn.functional.avg_pool3d(
                input_reshaped,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            
            return result.view(result.size(2), result.size(3), result.size(4))
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for 3D average pooling.
        
        Returns:
            List of test case dictionaries with varying sizes
        """
        test_configs = [
            (192, 192, 192, 5, 2, 2),
            (224, 224, 224, 7, 3, 3),
            (512, 512, 512, 3, 3, 1),
            (784, 784, 784, 5, 2, 2),
            (1024, 1024, 1024, 3, 3, 1),
        ]
        
        test_cases = []
        for h, w, d, k, s, p in test_configs:
            seed = Problem.get_seed(f"{self.name}_H={h}_W={w}_D={d}_K={k}_S={s}_P={p}")
            test_cases.append({
                "name": f"H={h}, W={w}, D={d}, K={k}, S={s}, P={p}",
                "height": h,
                "width": w,
                "depth": d,
                "kernel_size": k,
                "stride": s,
                "padding": p,
                "create_inputs": lambda h=h, w=w, d=d, k=k, s=s, p=p, seed=seed, dtype=dtype: (
                    *(lambda g: (
                        torch.rand((h, w, d), device="cuda", dtype=dtype, generator=g) * 2.0 - 1.0,  # uniform [-1, 1]
                    ))(torch.Generator(device="cuda").manual_seed(seed)),
                    k, s, p
                )
            })
        return test_cases
    
    def generate_sample(self, dtype: torch.dtype = torch.float32) -> List[Dict[str, Any]]:
        """
        Generate a single sample test case for debugging or interactive runs.
        
        Returns:
            A list containing a single test case dictionary
        """
        h, w, d, k, s, p = (4, 4, 4, 3, 1, 1) # Sample configuration
        return {
            "name": f"H={h}, W={w}, D={d}, K={k}, S={s}, P={p}",
            "height": h,
            "width": w,
            "depth": d,
            "kernel_size": k,
            "stride": s,
            "padding": p,
            "create_inputs": lambda h=h, w=w, d=d, k=k, s=s, p=p: (
                torch.arange(1, h * w * d + 1, device="cuda", dtype=dtype).float().view(h, w, d), # Sequential input
                k, s, p
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
        is_close = torch.allclose(actual_output, expected_output, rtol=2e-4, atol=1e-5)
        
        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            max_diff = torch.max(torch.abs(diff)).item()
            mean_diff = torch.mean(torch.abs(diff)).item()
            
            # Find indices of largest differences
            flat_diff = diff.flatten()
            _, top_indices = torch.topk(torch.abs(flat_diff), min(5, flat_diff.numel()))
            
            # Convert flat indices back to 3D coordinates
            h, w, d = expected_output.shape
            sample_diffs = {}
            for i, idx in enumerate(top_indices):
                row = idx.item() // (w * d)
                col = (idx.item() // d) % w
                depth = idx.item() % d
                sample_diffs[f"({row}, {col}, {depth})"] = {
                    "expected": expected_output[row, col, depth].item(),
                    "actual": actual_output[row, col, depth].item(),
                    "diff": diff[row, col, depth].item()
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
                ctypes.c_size_t,                 # width (W)
                ctypes.c_size_t,                 # depth (D)
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
        H = test_case["height"]
        W = test_case["width"]
        D = test_case["depth"]
        K = test_case["kernel_size"]
        S = test_case["stride"]
        P = test_case["padding"]

        
        # Calculate output dimensions
        H_out = ((H + 2 * P - K) // S) + 1
        W_out = ((W + 2 * P - K) // S) + 1
        D_out = ((D + 2 * P - K) // S) + 1
        # Each output element requires:
        # - (K*K*K - 1) additions for summing the window
        # - 1 division for computing the average
        ops_per_output = K * K * K - 1 + 1
        
        # Total FLOPs for the entire output
        return H_out * W_out * D_out * ops_per_output
    
    def get_mem(self, test_case: Dict[str, Any]) -> int:
        """
        Get the memory usage for the problem. Assumed to be all in DRAM
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            Memory usage in bytes
        """
        H = test_case["height"]
        W = test_case["width"]
        D = test_case["depth"]
        K = test_case["kernel_size"]
        S = test_case["stride"]
        P = test_case["padding"]
        
        # Calculate output dimensions
        H_out = ((H + 2 * P - K) // S) + 1
        W_out = ((W + 2 * P - K) // S) + 1
        D_out = ((D + 2 * P - K) // S) + 1
        
        # Input: H*W*D elements, Output: H_out*W_out*D_out elements
        dtype_bytes = 4  # 4 bytes per float32 element
        return (H * W * D + H_out * W_out * D_out) * dtype_bytes
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the image height H and width W
        """
        return [
            test_case["height"],
            test_case["width"],
            test_case["depth"]
        ]
