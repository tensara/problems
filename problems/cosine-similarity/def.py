import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem

class cosine_similarity(Problem):
    """Cosine Similarity problem."""
    
    is_exact = False
    
    def __init__(self):
        super().__init__(
            name="cosine-similarity"
        )
    
    def reference_solution(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of element-wise Cosine Similarity.
        
        Args:
            predictions: Predictions tensor of shape (N, D)
            targets: Targets tensor of shape (N, D)
            
        Returns:
            Negative cosine similarity tensor of shape (N,)
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            return 1 - torch.nn.functional.cosine_similarity(predictions, targets, dim=1)
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for Cosine Similarity.
        
        Returns:
            List of test case dictionaries with varying sizes.
        """
        batch_size = 128
        input_shape = (4096, )
        
        tensor_sizes = [
            4096,         # 4K vectors
            8192,         # 8K vectors
            10240,        # 10K vectors
            16384,        # 16K vectors
        ]
        
        test_cases = []
        for n in tensor_sizes:
            name = f"N={n}, D={input_shape[0]}"
            seed = Problem.get_seed(f"{self.name}_{name}_{(n, input_shape[0])}")
            test_cases.append({
                "name": name,
                "n": n,
                "d": input_shape[0],
                "create_inputs": lambda n=n, d=input_shape[0], seed=seed, dtype=dtype: (
                    *(lambda g: (
                        torch.randn(n, d, device="cuda", dtype=dtype, generator=g),  # predictions
                        torch.randn(n, d, device="cuda", dtype=dtype, generator=g)   # targets
                    ))(torch.Generator(device="cuda").manual_seed(seed)),
                )
            })
        
        return test_cases
    
    def generate_sample(self, dtype: torch.dtype = torch.float32) -> List[Dict[str, Any]]:
        """
        Generate a single sample test case for debugging or interactive runs.
        
        Returns:
            A list containing a single test case dictionary
        """
        n, d = (8, 8)
        return {
            "name": f"N={n}, D={d}",
            "n": n,
            "d": d,
            "create_inputs": lambda n=n, d=d: (
                torch.arange(1, n * d + 1, device="cuda", dtype=dtype).float().view(n, d),
                torch.flip(torch.arange(1, n * d + 1, device="cuda", dtype=dtype).float().view(n, d), dims=[1])
            )
        }
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the Cosine Similarity result is correct.
        
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
        Get the function signature for the Cosine Similarity solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        # Corresponds to parameters in problem.md
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # predictions
                ctypes.POINTER(ctypes.c_float),  # targets
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_size_t,                 # n (number of vectors)
                ctypes.c_size_t                  # d (dimension of each vector)
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
        D = test_case["d"]  # Dimension of each vector
        
        # For each of the N vector pairs:
        # 1. Dot product: predictions[i] · targets[i]  (2*D FLOPs: D mults, D-1 adds)
        # 2. Norm of predictions[i]:
        #    - Square each element (D FLOPs)
        #    - Sum squares (D-1 FLOPs)
        #    - Square root (1 FLOP)
        # 3. Norm of targets[i]: same as above (2*D FLOPs)
        # 4. Division: dot_product / (norm_pred * norm_targ) (2 FLOPs: mult, div)
        # 5. Subtraction: 1 - result (1 FLOP)
        
        # Total per vector pair: approximately 5*D + 3 FLOPs
        return N * (5 * D + 3)
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the number of vectors N.
        """
        N = test_case["n"]
        D = test_case["d"]
        return [N, D]
