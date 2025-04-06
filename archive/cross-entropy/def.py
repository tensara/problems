import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class cross_entropy(Problem):
    """Cross Entropy problem."""
    
    def __init__(self):
        super().__init__(
            name="cross-entropy"
        )
    
    def reference_solution(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of Cross Entropy Loss.
        
        Args:
            predictions: Predictions tensor of shape (N, C) where C is the number of classes
            targets: Targets tensor of shape (N,) containing class indices
            
        Returns:
            Cross entropy loss tensor of shape (N,)
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            return torch.nn.functional.cross_entropy(predictions, targets, reduction='none')
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for Cross Entropy.
        
        Returns:
            List of test case dictionaries with varying sizes.
        """
        test_configs = [
            (1024, 10),      # 1K samples, 10 classes
            (4096, 100),     # 4K samples, 100 classes  
            (16384, 50),     # 16K samples, 50 classes
            (65536, 25),     # 64K samples, 25 classes
        ]
        
        test_cases = []
        for n, num_classes in test_configs:
            test_cases.append({
                "name": f"N={n}, C={num_classes}",
                "n": n,
                "num_classes": num_classes,
                "create_inputs": lambda n=n, c=num_classes: (
                    torch.randn(n, c, device="cuda", dtype=dtype),  # predictions
                    torch.randint(0, c, (n,), device="cuda")        # targets
                )
            })
        
        return test_cases
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the Cross Entropy result is correct.
        
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

        is_close = torch.allclose(actual_output, expected_output, rtol=1e-5, atol=1e-5)
        
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
                sample_diffs[f"{idx.item()}"] = {
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
        Get the function signature for the Cross Entropy solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        # Corresponds to parameters in problem.md
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # predictions
                ctypes.POINTER(ctypes.c_int),    # targets (integer class indices)
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_size_t,                 # n (number of samples)
                ctypes.c_size_t                  # c (number of classes)
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
        C = test_case["num_classes"]  # Number of classes varies per test case
        
        # For each of the N samples:
        # 1. Compute max for numerical stability (C FLOPs)
        # 2. Subtract max from each logit (C FLOPs)
        # 3. Compute exp for each class (C FLOPs)
        # 4. Sum of exps (C-1 FLOPs)
        # 5. Log of sum (1 FLOP)
        # 6. Compute log probability of the correct class (2 FLOPs)
        # 7. Negate the result (1 FLOP)
        
        # Total per sample: approximately 4*C + 2 FLOPs
        return N * (4 * C + 2)
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the number of samples N.
        """
        N = test_case["n"]
        C = test_case["num_classes"]
        return [N, C]
