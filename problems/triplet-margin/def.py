import torch
import torch.nn as nn
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem 

class triplet_margin(Problem):
    """Triplet Margin Loss problem."""

    def __init__(self):
        super().__init__(
            name="triplet-margin"
        )
        self.margin = 1.0  # Standard margin for TripletMarginLoss

    def reference_solution(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of Triplet Margin Loss.

        Args:
            anchor (torch.Tensor): Anchor points, shape (batch_size, embedding_dim)
            positive (torch.Tensor): Positive examples, shape (batch_size, embedding_dim)
            negative (torch.Tensor): Negative examples, shape (batch_size, embedding_dim)

        Returns:
            torch.Tensor: Triplet loss value (scalar)
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            # Use PyTorch's built-in TripletMarginLoss
            loss_fn = nn.TripletMarginLoss(margin=self.margin)
            
            # Move to the same device as inputs
            loss_fn = loss_fn.to(anchor.device)
            
            # Calculate the loss
            loss = loss_fn(anchor, positive, negative)
            
            return loss

    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for Triplet Margin Loss.

        Returns:
            List of test case dictionaries with varying sizes
        """
        
        # Define configurations: (batch_size, embedding_dim)
        test_configs = [
            (128, 4096),      
            (256, 8192),     
            (256, 16384),    
            (512, 8192),     
            (1024, 1024)     
        ]

        return [
            {
                "name": f"batch={batch}, embedding_dim={embedding_dim}",
                "batch": batch,
                "embedding_dim": embedding_dim,
                "create_inputs": lambda batch=batch, embedding_dim=embedding_dim: (
                    torch.randn(batch, embedding_dim, device="cuda", dtype=dtype),  # Anchor
                    torch.randn(batch, embedding_dim, device="cuda", dtype=dtype),  # Positive
                    torch.randn(batch, embedding_dim, device="cuda", dtype=dtype)   # Negative
                )
            }
            for batch, embedding_dim in test_configs
        ]

    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the Triplet Margin Loss result is correct.

        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution

        Returns:
            Tuple of (is_correct, debug_info)
        """
        # Use appropriate tolerance based on precision
        rtol = 1e-3 if dtype == torch.float16 else 1e-4
        atol = 1e-3 if dtype == torch.float16 else 1e-5
        
        is_close = torch.allclose(actual_output, expected_output, rtol=rtol, atol=atol)
        
        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            max_diff = torch.max(torch.abs(diff)).item()

            debug_info = {
                "max_difference": max_diff,
            }
        
        return is_close, debug_info

    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the Triplet Margin Loss solution.

        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # anchor (input)
                ctypes.POINTER(ctypes.c_float),  # positive (input)
                ctypes.POINTER(ctypes.c_float),  # negative (input)
                ctypes.POINTER(ctypes.c_float),  # loss (output)
                ctypes.c_size_t,                 # batch_size
                ctypes.c_size_t,                 # embedding_dim
                ctypes.c_float,                  # margin
            ],
            "restype": None
        }

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the approximate number of floating point operations for Triplet Margin Loss.

        Args:
            test_case: The test case dictionary

        Returns:
            Number of floating point operations
        """
        batch = test_case["batch"]
        embedding_dim = test_case["embedding_dim"]
        
        # FLOPs calculation:
        # 1. Calculate anchor-positive distance: 
        #    - Subtraction: batch * embedding_dim
        #    - Square: batch * embedding_dim
        #    - Sum: batch * (embedding_dim-1)
        #    - Square root: batch * 5 (approx cost of sqrt)
        
        # 2. Calculate anchor-negative distance (same as above):
        #    - Subtraction: batch * embedding_dim
        #    - Square: batch * embedding_dim
        #    - Sum: batch * (embedding_dim-1)
        #    - Square root: batch * 5 (approx cost of sqrt)
        
        # 3. Calculate loss:
        #    - Subtraction (d_pos - d_neg + margin): batch * 2
        #    - ReLU (max(0, x)): batch * 1
        #    - Mean: batch - 1
        
        flops_per_distance = embedding_dim + embedding_dim + (embedding_dim - 1) + 5
        flops_for_distances = 2 * batch * flops_per_distance  # For both positive and negative
        
        flops_for_loss = batch * 3 + (batch - 1)
        
        total_flops = flops_for_distances + flops_for_loss
        
        return int(total_flops)

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.

        Args:
            test_case: The test case dictionary

        Returns:
            List containing batch_size, embedding_dim, margin
        """
        batch = test_case["batch"]
        embedding_dim = test_case["embedding_dim"]
            
        return [batch, embedding_dim, self.margin]
