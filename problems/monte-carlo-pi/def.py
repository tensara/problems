import torch
import ctypes
import math
from typing import List, Dict, Tuple, Any

from problem import Problem


class monte_carlo_pi(Problem):
    """Monte Carlo pi estimation problem."""
    
    def __init__(self):
        super().__init__(
            name="monte-carlo-pi"
        )
    
    def reference_solution(self, random_x: torch.Tensor, random_y: torch.Tensor, n_samples: int) -> torch.Tensor:
        """
        PyTorch implementation of Monte Carlo pi estimation.
        
        Args:
            random_x: Random x coordinates in [0, 1]
            random_y: Random y coordinates in [0, 1]
            n_samples: Number of samples to use
            
        Returns:
            Estimated value of pi
        """
        with torch.no_grad():
            # Count points inside unit circle (x^2 + y^2 <= 1)
            distances_squared = random_x * random_x + random_y * random_y
            inside_circle = (distances_squared <= 1.0).float()
            points_inside = torch.sum(inside_circle)
            
            # Pi estimation: 4 * (points inside circle) / (total points)
            pi_estimate = 4.0 * points_inside / n_samples
            
            return pi_estimate.unsqueeze(0)  # Return as tensor with shape (1,)
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for Monte Carlo pi estimation.
        
        Returns:
            List of test case dictionaries with varying sample counts
        """
        test_cases = [
            {
                "name": "1M_samples",
                "n_samples": 1_000_000,
                "create_inputs": lambda n=1_000_000: self._create_random_points(n, dtype)
            },
            {
                "name": "10M_samples",
                "n_samples": 10_000_000,
                "create_inputs": lambda n=10_000_000: self._create_random_points(n, dtype)
            },
            {
                "name": "50M_samples",
                "n_samples": 50_000_000,
                "create_inputs": lambda n=50_000_000: self._create_random_points(n, dtype)
            },
            {
                "name": "100M_samples",
                "n_samples": 100_000_000,
                "create_inputs": lambda n=100_000_000: self._create_random_points(n, dtype)
            }
        ]
        
        return test_cases
    
    def _create_random_points(self, n_samples: int, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Helper to create random points in unit square."""
        # Generate random points in [0, 1] x [0, 1]
        random_x = torch.rand(n_samples, device="cuda", dtype=dtype)
        random_y = torch.rand(n_samples, device="cuda", dtype=dtype)
        
        return (random_x, random_y, n_samples)
    
    def generate_sample(self, dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
        """
        Generate a single sample test case for debugging.
        
        Returns:
            A dictionary containing a single test case
        """
        return {
            "name": "Sample (n=1000)",
            "n_samples": 1000,
            "create_inputs": lambda: self._create_sample_points(dtype)
        }
    
    def _create_sample_points(self, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Create a small sample for debugging with known seed."""
        # Use deterministic values for reproducible testing
        torch.manual_seed(42)
        n_samples = 1000
        random_x = torch.rand(n_samples, device="cuda", dtype=dtype)
        random_y = torch.rand(n_samples, device="cuda", dtype=dtype)
        
        return (random_x, random_y, n_samples)

    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the Monte Carlo pi estimation result is correct.
        
        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            
        Returns:
            Tuple of (is_correct, debug_info)
        """
        # For Monte Carlo methods, we expect some variance, but results should be close
        is_close = torch.allclose(actual_output, expected_output, rtol=1e-6, atol=1e-6)
        
        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            max_diff = torch.max(torch.abs(diff)).item()
            
            # Calculate error from true pi
            true_pi = math.pi
            expected_error = abs(expected_output.item() - true_pi)
            actual_error = abs(actual_output.item() - true_pi)
            
            debug_info = {
                "difference": max_diff,
                "expected_pi": expected_output.item(),
                "actual_pi": actual_output.item(),
                "true_pi": true_pi,
                "expected_error_from_true_pi": expected_error,
                "actual_error_from_true_pi": actual_error
            }
        
        return is_close, debug_info
    
    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the Monte Carlo pi estimation solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # random_x
                ctypes.POINTER(ctypes.c_float),  # random_y
                ctypes.POINTER(ctypes.c_float),  # pi_estimate (output)
                ctypes.c_size_t                  # n_samples
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
        n_samples = test_case["n_samples"]
        
        # For each sample point:
        # - 1 multiplication for x^2
        # - 1 multiplication for y^2  
        # - 1 addition for x^2 + y^2
        # - 1 comparison (x^2 + y^2 <= 1)
        # - 1 addition to accumulate count
        # Total: 4 operations per sample
        # Final: 1 multiplication by 4, 1 division by n_samples
        # Total: 4 * n_samples + 2
        return 4 * n_samples + 2
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the number of samples
        """
        return [test_case["n_samples"]] 