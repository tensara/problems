import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class matmul_swish(Problem):
    """Matrix multiplication with Swish activation problem."""
    
    def __init__(self):
        super().__init__(
            name="matmul-swish"
        )
    
    def reference_solution(self, input_matrix: torch.Tensor, weight_matrix: torch.Tensor, 
                         bias: torch.Tensor, scaling_factor: float) -> torch.Tensor:
        """
        PyTorch implementation of matrix multiplication with Swish activation.
        
        Args:
            input_matrix: Input tensor of shape (batch_size, in_features)
            weight_matrix: Weight tensor of shape (out_features, in_features)
            bias: Bias tensor of shape (out_features,)
            scaling_factor: Scaling factor to apply after Swish activation
            
        Returns:
            Result of matrix multiplication with Swish activation and scaling
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            # Linear transformation
            z = torch.matmul(input_matrix, weight_matrix.t()) + bias
            
            # Swish activation: x * sigmoid(x)
            output = z * torch.sigmoid(z)
            
            # Apply scaling
            output = output * scaling_factor
            
            return output
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for matrix multiplication with Swish activation.
        
        Returns:
            List of test case dictionaries with varying sizes
        """
        test_configs = [
            (128, 1024, 512, 2.0),    # Standard size
            (256, 2048, 1024, 1.5),   # Larger size
            (64, 512, 256, 0.5),      # Smaller size
            (512, 4096, 2048, 3.0),   # Very large size
            (32, 256, 128, 1.0),      # Very small size
        ]
        
        return [
            {
                "name": f"B={b}, In={i}, Out={o}, Scale={s}",
                "batch_size": b,
                "in_features": i,
                "out_features": o,
                "scaling_factor": s,
                "create_inputs": lambda b=b, i=i, o=o, s=s: (
                    torch.randn((b, i), device="cuda", dtype=dtype) * 0.1,  # input_matrix
                    torch.randn((o, i), device="cuda", dtype=dtype) * 0.1,  # weight_matrix
                    torch.randn(o, device="cuda", dtype=dtype) * 0.1,       # bias
                    s  # scaling_factor
                )
            }
            for b, i, o, s in test_configs
        ]
    
    def generate_sample(self, dtype: torch.dtype = torch.float32) -> List[Dict[str, Any]]:
        """
        Generate sample test cases for matrix multiplication with Swish activation.

        Returns:
            List of sample test case dictionaries with predictable inputs.
        """
        sample_configs = [
            {
                "name": "sample_small",
                "batch_size": 2,
                "in_features": 3,
                "out_features": 4,
                "scaling_factor": 1.5
            },
            {
                "name": "sample_medium",
                "batch_size": 3,
                "in_features": 4,
                "out_features": 2,
                "scaling_factor": 0.8
            }
        ]

        return [
            {
                "name": config["name"],
                "batch_size": config["batch_size"],
                "in_features": config["in_features"],
                "out_features": config["out_features"],
                "scaling_factor": config["scaling_factor"],
                "create_inputs": lambda b=config["batch_size"], i=config["in_features"], o=config["out_features"], s=config["scaling_factor"], dtype_val=dtype: (
                    torch.arange(b * i, device="cuda", dtype=dtype_val).reshape(b, i) / (b * i),
                    torch.arange(o * i, device="cuda", dtype=dtype_val).reshape(o, i) / (o * i),
                    torch.arange(o, device="cuda", dtype=dtype_val) / o,
                    s
                )
            }
            for config in sample_configs
        ]
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the result is correct.
        
        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            
        Returns:
            Tuple of (is_correct, debug_info)
        """
        is_close = torch.allclose(actual_output, expected_output, rtol=1e-4, atol=1e-4)
        
        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            max_diff = torch.max(torch.abs(diff)).item()
            mean_diff = torch.mean(torch.abs(diff)).item()
            
            # Find indices of largest differences
            flat_diff = diff.flatten()
            _, top_indices = torch.topk(torch.abs(flat_diff), min(5, flat_diff.numel()))
            
            # Convert flat indices back to 2D coordinates
            b, o = expected_output.shape
            sample_diffs = {}
            for i, idx in enumerate(top_indices):
                row = idx.item() // o
                col = idx.item() % o
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
        Get the function signature for the solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # input_matrix
                ctypes.POINTER(ctypes.c_float),  # weight_matrix
                ctypes.POINTER(ctypes.c_float),  # bias
                ctypes.c_float,                  # scaling_factor
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_size_t,                 # batch_size
                ctypes.c_size_t,                 # in_features
                ctypes.c_size_t,                 # out_features
            ],
            "restype": None
        }
    
    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the number of floating point operations for the problem.
        
        Args:
            test_case: The test case dictionary
            
        For each output element we have:
        - in_features multiplications and (in_features-1) additions for matmul
        - 1 addition for bias
        - ~4 FLOPs for sigmoid (approximation)
        - 2 multiplications (one for Swish, one for scaling)
        
        Returns:
            Number of floating point operations
        """
        B = test_case["batch_size"]
        I = test_case["in_features"]
        O = test_case["out_features"]
        
        # For each output element:
        flops_per_element = (
            2 * I +        # matmul (I muls and I-1 adds)
            1 +           # bias add
            4 +           # sigmoid
            2            # Swish mul and scaling mul
        )
        
        # Total FLOPs for all elements
        return B * O * flops_per_element
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the batch_size, in_features, and out_features
        """
        return [
            test_case["batch_size"],
            test_case["in_features"],
            test_case["out_features"],
        ]
