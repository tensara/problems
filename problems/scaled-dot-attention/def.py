import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class scaled_dot_attention(Problem):
    """Scaled Dot-Product Attention problem."""
    
    is_exact = False
    
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
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=query.dtype):
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
            (8, 16, 512, 64),  
            (16, 32, 256, 64), 
            (4, 32, 1024, 32),  
            (32, 8, 512, 128),  
            (8, 16, 512, 256),
            (8, 16, 2048, 64)  
        ]
        
        test_cases = []
        for b, h, s, e in test_configs:
            name = f"Batch={b}, Heads={h}, Seq_len={s}, Embed_dim={e}"
            seed = Problem.get_seed(f"{self.name}_{name}_{(b, h, s, e)}")
            test_cases.append({
                "name": name,
                "batch": b,
                "heads": h,
                "seq_len": s,
                "embed_dim": e,
                "create_inputs": lambda b=b, h=h, s=s, e=e, seed=seed, dtype=dtype: (
                    (lambda g: (
                        torch.randn((b, h, s, e), device="cuda", dtype=dtype, generator=g) * 0.01,  # query
                        torch.randn((b, h, s, e), device="cuda", dtype=dtype, generator=g) * 0.01,  # key
                        torch.randn((b, h, s, e), device="cuda", dtype=dtype, generator=g) * 0.01   # value
                    ))(torch.Generator(device="cuda").manual_seed(seed))
                )
            })
        return test_cases
    
    def generate_sample(self, dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
        """
        Generate sample test case for scaled dot-product attention with predictable inputs.

        Returns:
            Dictionary containing the sample test case.
        """
        batch, heads, seq_len, embed_dim = 1, 2, 2, 2
        return {
            "name": "1x2x2x2_sample",
            "batch": batch,
            "heads": heads,
            "seq_len": seq_len,
            "embed_dim": embed_dim,
            "create_inputs": lambda: (
                torch.tensor([
                    [ 
                        [
                            [1.0, 2.0],
                            [3.0, 4.0]
                        ],
                        [
                            [0.1, 0.2],
                            [0.3, 0.4]
                        ]
                    ]
                ], device="cuda", dtype=dtype),
                torch.tensor([
                    [
                        [
                            [1.0, 0.0],
                            [0.0, 1.0]
                        ],
                        [
                            [0.5, 1.5],
                            [1.5, 0.5]
                        ]
                    ]
                ], device="cuda", dtype=dtype),
                torch.tensor([
                    [ 
                        [
                            [10.0, 20.0],
                            [30.0, 40.0]
                        ],
                        [
                            [5.0, 15.0],
                            [25.0, 35.0]
                        ]
                    ]
                ], device="cuda", dtype=dtype)
            )
        }
    
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
        is_close = torch.allclose(actual_output, expected_output, rtol=2e-2, atol=5e-3)
        
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
        # 1. Matrix multiplication QK^T: 2 * b*h*s*s*e operations
        # 2. Scaling by 1/sqrt(e): b*h*s*s operations
        # 3. Softmax:
        #    - Find max: b*h*s*s operations
        #    - Subtract max and exp: 2*b*h*s*s operations
        #    - Normalization (sum and divide): 2*b*h*s*s operations
        #    - Overall: 5*b*h*s*s operations
        # 4. Matrix multiplication with V: 2 * b*h*s*s*e operations
        
        # Total FLOPs: 4*b*h*s*s*e + 5*b*h*s*s
        # For simplicity and to align with standard calculation methods, 
        # we'll use a more straightforward approximation focusing on the most expensive operations
        
        return (4 * b * h * s * s * e) + (5 * b * h * s * s)
    
    def get_mem(self, test_case: Dict[str, Any]) -> int:
        """
        Get the memory usage for the problem. Assumed to be all in DRAM
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            Memory usage in bytes
        """
        b = test_case["batch"]
        h = test_case["heads"]
        s = test_case["seq_len"]
        e = test_case["embed_dim"]
        
        # Naive scaled dot-product attention:
        # 1. Read Q, K for QK^T → 2 * b*h*s*e
        # 2. Write QK^T → b*h*s*s
        # 3. Read QK^T (scaling) → b*h*s*s
        # 4. Write scaled → b*h*s*s
        # 5. Softmax: read scaled (max) → b*h*s*s, read scaled (exp) → b*h*s*s, write exp → b*h*s*s, 
        #    read exp (norm) → b*h*s*s, write softmax → b*h*s*s = 5 * b*h*s*s
        # 6. Read softmax, V for matmul → b*h*s*s + b*h*s*e
        # 7. Write output → b*h*s*e
        
        dtype_bytes = 4  # 4 bytes per float32 element
        return (2 * b * h * s * e +  # Q, K reads
                8 * b * h * s * s +  # QK^T, scaled, softmax intermediates (1+1+5+1)
                b * h * s * e +      # V read
                b * h * s * e) * dtype_bytes  # output write
    
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
