import torch
import ctypes
from typing import List, Dict, Tuple, Any
import math

from problem import Problem


class array_sort(Problem):
    """General array sorting problem."""
    
    def __init__(self):
        super().__init__(
            name="array-sort"
        )
    
    def reference_solution(self, input_array: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of array sorting.
        
        Args:
            input_array: Input array of shape (n,)
            
        Returns:
            Sorted array of shape (n,)
        """
        with torch.no_grad():
            sorted_array = torch.sort(input_array)[0]
            return sorted_array
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for array sorting.
        
        Returns:
            List of test case dictionaries with varying array sizes
        """
        array_sizes = [
            50000,
            100000,
            500000,
            1000000,
            5000000,
            10000000,
            20000000,
            50000000
        ]
        
        test_cases = []
        for size in array_sizes:
            test_cases.append({
                "name": f"size_{size}",
                "size": size,
                "create_inputs": lambda s=size: (
                    torch.randint(0, 1000, (s,), device="cuda", dtype=dtype),
                )
            })
        
        return test_cases
    
    def generate_sample(self, dtype: torch.dtype = torch.float32) -> List[Dict[str, Any]]:
        """
        Generate a single sample test case for debugging or interactive runs.
        
        Returns:
            A list containing a single test case dictionary
        """
        size = 16
        return {
            "name": "Sample (size=16)",
            "size": size,
            "create_inputs": lambda s=size: (
                torch.randint(0, 1000, (s,), device="cuda", dtype=dtype),
            )
        }

    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the sorting result is correct.
        
        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            
        Returns:
            Tuple of (is_correct, debug_info)
        """
        is_close = torch.allclose(actual_output, expected_output, rtol=1e-5, atol=1e-5)
        
        debug_info = {}
        if not is_close:
            # Check if array is sorted
            is_sorted = torch.all(actual_output[:-1] <= actual_output[1:])
            
            # Find first mismatch
            diff = actual_output - expected_output
            nonzero_diff = torch.nonzero(torch.abs(diff) > 1e-5)
            
            sample_diffs = {}
            if nonzero_diff.size(0) > 0:
                for i in range(min(5, nonzero_diff.size(0))):
                    idx = nonzero_diff[i, 0].item()
                    sample_diffs[f"index_{idx}"] = {
                        "expected": expected_output[idx].item(),
                        "actual": actual_output[idx].item(),
                        "diff": diff[idx].item()
                    }
                        
            debug_info = {
                "is_sorted": is_sorted.item(),
                "sample_differences": sample_diffs,
                "total_mismatches": nonzero_diff.size(0)
            }
        
        return is_close, debug_info
    
    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the array sorting solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_int),  # input_array
                ctypes.POINTER(ctypes.c_int),  # output_array
                ctypes.c_size_t,                 # size
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
        size = test_case["size"]
        
        # General sorting algorithms vary in complexity:
        # - Best case (already sorted): O(n) comparisons
        # - Average case: O(n log n) comparisons for efficient algorithms
        # - Worst case: O(n^2) for simple algorithms, O(n log n) for advanced ones
        # We'll use O(n log n) as a reasonable estimate for efficient sorting
        return int(size * math.log2(size) * 3)  # 3 FLOPs per comparison (compare + potential swap operations)
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the array size
        """
        size = test_case["size"]
        return [size] 