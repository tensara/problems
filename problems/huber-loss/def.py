import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class huber_loss(Problem):
    """Huber Loss (Smooth L1 Loss) problem."""
    
    is_exact = False
    
    def __init__(self):
        super().__init__(
            name="huber-loss"
        )
    
    def reference_solution(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of element-wise Huber Loss (Smooth L1 Loss).
        
        Args:
            predictions: Predictions tensor of shape (N,)
            targets: Targets tensor of shape (N,)
            
        Returns:
            Element-wise Huber loss tensor of shape (N,)
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=predictions.dtype):
            # Use reduction='none' to get element-wise loss
            return torch.nn.functional.smooth_l1_loss(predictions, targets, reduction='none', beta=1.0)
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for Huber Loss.
        
        Returns:
            List of test case dictionaries with varying sizes.
        """
        tensor_sizes = [
            1048576,      # 1M elements
            4194304,      # 4M elements
            16777216,     # 16M elements
            67108864,     # 64M elements
        ]
        
        test_cases = []
        for n in tensor_sizes:
            seed = Problem.get_seed(f"{self.name}_N={n}")
            test_cases.append({
                "name": f"N={n}",
                "n": n,
                "create_inputs": lambda n=n, seed=seed, dtype=dtype: (
                    (lambda g: (
                        torch.randn(n, device="cuda", dtype=dtype, generator=g), # predictions
                        torch.randn(n, device="cuda", dtype=dtype, generator=g)  # targets
                    ))(torch.Generator(device="cuda").manual_seed(seed))
                )
            })
        
        return test_cases
    
    def generate_sample(self, dtype: torch.dtype = torch.float32) -> List[Dict[str, Any]]:
        """
        Generate a single sample test case for debugging or interactive runs.
        
        Returns:
            A list containing a single test case dictionary
        """
        n = 8 
        return {
            "name": f"Sample N={n}",
            "n": n,
            "create_inputs": lambda n=n: (
                torch.tensor([0.2, 0.8, 1.0, 1.5, -0.3, -0.9, -1.2, 2.0], device="cuda", dtype=dtype),
                torch.tensor([0.0, 0.5, 0.3, 2.0, 0.1, -0.2, -2.0, 0.5], device="cuda", dtype=dtype) 
            )
        }
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the Huber Loss result is correct.
        
        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            
        Returns:
            Tuple of (is_correct, debug_info)
        """
        # Ensure output has the correct shape
        if actual_output.shape != expected_output.shape:
             return False, {
                 "message": "Output shape mismatch, expected: {}, actual: {}".format(expected_output.shape, actual_output.shape),
             }

        is_close = torch.allclose(actual_output, expected_output, rtol=2e-4, atol=1e-4)
        
        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            max_diff = torch.max(torch.abs(diff)).item()
            mean_diff = torch.mean(torch.abs(diff)).item()
            
            # Find indices of largest differences
            flat_diff = diff.flatten()
            _, top_indices = torch.topk(torch.abs(flat_diff), min(5, flat_diff.numel()))
            
            n = expected_output.numel()
            sample_diffs = {}
            for i, idx in enumerate(top_indices):
                sample_diffs[f"index {idx.item()}"] = {
                    "expected": expected_output.flatten()[idx].item(),
                    "actual": actual_output.flatten()[idx].item(),
                    "diff": diff.flatten()[idx].item()
                }
                        
            debug_info = {
                "max_difference": max_diff,
                "mean_difference": mean_diff,
                "sample_differences": sample_diffs
            }
        
        return is_close, debug_info
    
    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the Huber Loss solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        # Corresponds to parameters in problem.md
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # predictions
                ctypes.POINTER(ctypes.c_float),  # targets
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_size_t                  # n (number of elements)
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
        N = test_case["n"]
        
        # N * ~5 FLOPs: For each element:
        # 1. diff = predictions[i] - targets[i] (1 FLOP)
        # 2. abs_diff = abs(diff)             (1 FLOP, approximate)
        # 3. Check condition: abs_diff < 1.0  (1 FLOP, comparison)
        # 4. Calculate result:
        #    - if true: 0.5 * diff * diff      (2 FLOPs: mult, mult)
        #    - if false: abs_diff - 0.5      (1 FLOP: sub)
        # Conservatively estimate as 5 FLOPs per element.
        return N * 5 
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the number of elements N.
        """
        N = test_case["n"]
        return [N]
