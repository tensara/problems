import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class matrix_power(Problem):
    """Matrix nth power problem."""
    
    def __init__(self):
        super().__init__(
            name="matrix-power"
        )
    
    def reference_solution(self, matrix_a: torch.Tensor, n: int) -> torch.Tensor:
        """
        PyTorch implementation of matrix nth power.
        
        Args:
            matrix_a: Input matrix of shape (N, N)
            n: Power to raise the matrix to
            
        Returns:
            Result of matrix^n of shape (N, N)
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            return torch.linalg.matrix_power(matrix_a, n)
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for matrix nth power.
        
        Returns:
            List of test case dictionaries with varying sizes and powers
        """
        matrix_sizes = [512, 1024, 2048]
        powers = [2, 4, 8]
        
        return [
            {
                "name": f"{size}x{size} power={power}",
                "size": size,
                "power": power,
                "create_inputs": lambda size=size, power=power: (
                    # Generate well-conditioned matrix with small values
                    torch.randn((size, size), device="cuda", dtype=dtype) * 0.01 + torch.eye(size, device="cuda", dtype=dtype) * 0.1,
                    power
                )
            }
            for size in matrix_sizes
            for power in powers
        ]
    
    def generate_sample(self, dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
        """
        Generate sample test cases for matrix nth power with predictable inputs.

        Returns:
            Sample test case dictionary.
        """
        size = 4
        power = 3
        return {
            "name": f"size={size}, power={power}",
            "size": size,
            "power": power,
            "create_inputs": lambda size_val=size, power_val=power, dtype_val=dtype: (
                # Create a simple 2x2 block diagonal matrix for predictability
                torch.block_diag(
                    torch.tensor([[1.1, 0.1], [0.1, 1.1]], device="cuda", dtype=dtype_val),
                    torch.tensor([[0.9, -0.1], [-0.1, 0.9]], device="cuda", dtype=dtype_val)
                ),
                power_val
            )
        }
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the matrix nth power result is correct.
        
        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            
        Returns:
            Tuple of (is_correct, debug_info)
        """
        is_close = torch.allclose(actual_output, expected_output, rtol=1e-4, atol=1e-3)
        
        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            max_diff = torch.max(torch.abs(diff)).item()
            mean_diff = torch.mean(torch.abs(diff)).item()
            
            # Find indices of largest differences
            flat_diff = diff.flatten()
            _, top_indices = torch.topk(torch.abs(flat_diff), min(5, flat_diff.numel()))
            
            # Convert flat indices back to 2D coordinates
            n = expected_output.shape[0]
            sample_diffs = {}
            for i, idx in enumerate(top_indices):
                row = idx.item() // n
                col = idx.item() % n
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
        Get the function signature for the matrix nth power solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # matrix_a
                ctypes.c_size_t,                 # power 
                ctypes.POINTER(ctypes.c_float), # output
                ctypes.c_size_t                 # size (N)
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
        # Extract dimension and power from test case
        N = test_case["size"]
        power = test_case["power"]
        
        # For matrix^n computation using repeated multiplication:
        # - We need (n-1) matrix multiplications
        # - Each matrix multiplication requires N^3 FLOPs:
        #   * For each of N^2 output elements
        #   * We compute dot product of N elements (N multiplications + N-1 additions)
        #   * Total: N^2 * (N + (N-1)) ≈ 2*N^3 FLOPs per matrix multiplication
        # - Total FLOPs: (n-1) * 2 * N^3
        
        if power <= 1:
            return 0  # No computation needed for power 0 or 1
        
        return (power - 1) * 2 * (N ** 3)
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the matrix size N
        """
        N = test_case["size"]
        return [N]