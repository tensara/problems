import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class scaled_dot_attention(Problem):
    """Scaled Dot-Product Attention problem."""
    
    def __init__(self):
        super().__init__(
            name="scaled-dot-attention"
        )
    
    def reference_solution(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of Scaled Dot-Product Attention.
        
        Args:
            query: Query tensor of shape (batch, heads, seq_len, embed_dim)
            key: Key tensor of shape (batch, heads, seq_len, embed_dim)
            value: Value tensor of shape (batch, heads, seq_len, embed_dim)
            
        Returns:
            Result of scaled dot-product attention
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            # Use PyTorch's built-in scaled dot product attention
            return torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for scaled dot-product attention.
        
        Returns:
            List of test case dictionaries with varying sizes
        """
        test_configs = [
            (1, 4, 512, 64),     # (batch, heads, seq_len, embed_dim)
            (4, 8, 1024, 64),
            (8, 16, 2048, 128),
            (16, 32, 512, 256),
            (32, 1, 128, 512),
            (2, 12, 4096, 64)
        ]
        
        return [
            {
                "name": f"Batch={b}, Heads={h}, Seq_len={s}, Embed_dim={e}",
                "batch": b,
                "heads": h,
                "seq_len": s,
                "embed_dim": e,
                "create_inputs": lambda b=b, h=h, s=s, e=e: (
                    torch.randn((b, h, s, e), device="cuda", dtype=dtype) * 0.01,  # query
                    torch.randn((b, h, s, e), device="cuda", dtype=dtype) * 0.01,  # key
                    torch.randn((b, h, s, e), device="cuda", dtype=dtype) * 0.01   # value
                )
            }
            for b, h, s, e in test_configs
        ]
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the attention result is correct.
        
        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            
        Returns:
            Tuple of (is_correct, debug_info)
        """
        is_close = torch.allclose(actual_output, expected_output, rtol=1e-3, atol=1e-3)
        
        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            max_diff = torch.max(torch.abs(diff)).item()
            mean_diff = torch.mean(torch.abs(diff)).item()
            
            # Sample differences from random positions
            b, h, s, e = expected_output.shape
            sample_diffs = {}
            
            # Choose a few random indices to sample
            indices = torch.randint(0, b, (3,), device=expected_output.device)
            for i, idx in enumerate(indices):
                h_idx = torch.randint(0, h, (1,)).item()
                s_idx = torch.randint(0, s, (1,)).item()
                e_idx = torch.randint(0, e, (1,)).item()
                
                sample_diffs[f"({idx.item()}, {h_idx}, {s_idx}, {e_idx})"] = {
                    "expected": expected_output[idx, h_idx, s_idx, e_idx].item(),
                    "actual": actual_output[idx, h_idx, s_idx, e_idx].item(),
                    "diff": diff[idx, h_idx, s_idx, e_idx].item()
                }
            
            debug_info = {
                "max_difference": max_diff,
                "mean_difference": mean_diff,
                "sample_differences": sample_diffs
            }
        
        return is_close, debug_info
    
    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the scaled dot-product attention solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # query
                ctypes.POINTER(ctypes.c_float),  # key
                ctypes.POINTER(ctypes.c_float),  # value
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_size_t,                 # batch
                ctypes.c_size_t,                 # heads
                ctypes.c_size_t,                 # seq_len
                ctypes.c_size_t                  # embed_dim
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
        # Extract dimensions from test case
        b = test_case["batch"]
        h = test_case["heads"]
        s = test_case["seq_len"]
        e = test_case["embed_dim"]
        
        # FLOPs calculation breakdown:
        # 1. Matrix multiplication QK^T: b*h*s*s*e operations
        # 2. Scaling by 1/sqrt(e): b*h*s*s operations
        # 3. Softmax:
        #    - Find max: b*h*s*s operations
        #    - Subtract max and exp: 2*b*h*s*s operations
        #    - Normalization (sum and divide): 2*b*h*s*s operations
        # 4. Matrix multiplication with V: b*h*s*s*e operations
        
        # Total FLOPs: 2*b*h*s*s*e + 5*b*h*s*s
        # For simplicity and to align with standard calculation methods, 
        # we'll use a more straightforward approximation focusing on the most expensive operations
        
        return 2 * b * h * s * s * e  # Dominated by the two matrix multiplications
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing batch size, number of heads, sequence length, and embedding dimension
        """
        batch = test_case["batch"]
        heads = test_case["heads"]
        seq_len = test_case["seq_len"]
        embed_dim = test_case["embed_dim"]
        return [batch, heads, seq_len, embed_dim]
