import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class kl_loss(Problem):
    """Kullback-Leibler Divergence problem."""
    
    def __init__(self):
        super().__init__(
            name="kl-loss"
        )
    
    def reference_solution(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of element-wise Kullback-Leibler Divergence.
        
        Args:
            predictions: Predictions tensor of shape (N,) representing a probability distribution
            targets: Targets tensor of shape (N,) representing a probability distribution
            
        Returns:
            Element-wise KL divergence tensor of shape (N,)
        """
        with torch.no_grad():
            # Add small epsilon to avoid numerical issues with log(0)
            eps = 1e-10
            pred_safe = predictions.clamp(min=eps)
            target_safe = targets.clamp(min=eps)
            
            # Compute element-wise KL divergence
            # Note: PyTorch's built-in KL div expects log-probabilities for predictions, 
            # but we're implementing the element-wise version directly
            element_wise_kl = target_safe * (torch.log(target_safe) - torch.log(pred_safe))
            
            # Zero out elements where target is 0 (by convention)
            element_wise_kl = torch.where(targets > 0, element_wise_kl, torch.zeros_like(element_wise_kl))
            
            return element_wise_kl
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for KL Divergence.
        
        Returns:
            List of test case dictionaries with varying sizes.
        """
        batch_size = 128
        input_shape = (4096, )
        
        tensor_sizes = [
            1048576,      # 1M elements
            4194304,      # 4M elements
            16777216,     # 16M elements
            67108864,     # 64M elements
        ]
        
        test_cases = []
        for n in tensor_sizes:
            name = f"N={n}"
            seed = Problem.get_seed(f"{self.name}_{name}_{(n,)}")
            test_cases.append({
                "name": name,
                "n": n,
                "create_inputs": lambda n=n, seed=seed, dtype=dtype: (
                    (lambda g: (
                        torch.rand(n, device="cuda", dtype=dtype, generator=g).softmax(dim=0),  # predictions
                        torch.rand(n, device="cuda", dtype=dtype, generator=g).softmax(dim=0)   # targets
                    ))(torch.Generator(device="cuda").manual_seed(seed))
                )
            })
        
        # Add special test case with sparse targets
        sparse_name = "Sparse_targets"
        sparse_n = 1048576
        sparse_seed = Problem.get_seed(f"{self.name}_{sparse_name}_{(sparse_n,)}")
        test_cases.append({
            "name": sparse_name,
            "n": sparse_n,
            "create_inputs": lambda seed=sparse_seed, dtype=dtype: (
                (lambda g: (
                    torch.rand(1048576, device="cuda", dtype=dtype, generator=g).softmax(dim=0),  # predictions
                    torch.zeros(1048576, device="cuda", dtype=dtype).scatter_(
                        0, torch.randint(0, 1048576, (1048576 // 10,), device="cuda", generator=g), 
                        torch.ones(1048576 // 10, device="cuda", dtype=dtype) * 10
                    ).softmax(dim=0)  # sparse targets
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
                torch.tensor([0.1, 0.2, 0.3, 0.4, 0.0, 0.6, 0.7, 0.8], device="cuda", dtype=dtype),
                torch.tensor([0.4, 0.3, 0.2, 0.1, 0.0, 0.2, 0.3, 0.4], device="cuda", dtype=dtype)
            )
        }
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the KL Divergence result is correct.
        
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

        # Use higher tolerance for KL divergence due to potential numerical issues
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
        Get the function signature for the KL Divergence solution.
        
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
        
        # N * ~7 FLOPs: For each element:
        # 1. Add epsilon to predictions and targets (~2 FLOPs)
        # 2. log(targets[i]) (1 FLOP)
        # 3. log(predictions[i]) (1 FLOP)
        # 4. log(targets[i]) - log(predictions[i]) (1 FLOP: sub)
        # 5. targets[i] * (log difference) (1 FLOP: mult)
        # 6. Check if targets[i] == 0 and conditionally set (1 FLOP)
        # Conservatively estimate as 7 FLOPs per element
        return N * 7 
    
    def get_mem(self, test_case: Dict[str, Any]) -> int:
        """
        Get the memory usage for the problem. Assumed to be all in DRAM
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            Memory usage in bytes
        """
        N = test_case["n"]
        
        # Naive KL divergence loss:
        # 1. Read predictions → N
        # 2. Read targets → N
        # 3. Write pred_eps = predictions + eps → N (materialized)
        # 4. Write targ_eps = targets + eps → N (materialized)
        # 5. Read pred_eps → N
        # 6. Write log_pred = log(pred_eps) → N (materialized)
        # 7. Read targ_eps → N
        # 8. Write log_targ = log(targ_eps) → N (materialized)
        # 9. Read log_pred, log_targ → 2*N
        # 10. Write log_diff = log_targ - log_pred → N (materialized)
        # 11. Read log_diff, targets → 2*N
        # 12. Write loss = targets * log_diff → N
        
        dtype_bytes = 4  # 4 bytes per float32 element
        return (N + N +          # read predictions, targets
                N + N +          # write pred_eps, targ_eps
                N + N +          # read pred_eps, write log_pred
                N + N +          # read targ_eps, write log_targ
                N + N +          # read log_pred, log_targ
                N + N +          # write and read log_diff
                N + N) * dtype_bytes  # read targets, write loss (output)
    
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
