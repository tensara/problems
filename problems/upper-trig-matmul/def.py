import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class upper_trig_matmul(Problem):
    """Upper triangular matrix multiplication problem."""
    
    is_exact = False
    
    def __init__(self):
        super().__init__(
            name="upper-trig-matmul"
        )
    
    def reference_solution(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of upper triangular matrix multiplication.
        Ensures inputs are upper triangular before multiplying.
        
        Args:
            A: First input matrix (expected to be upper triangular)
            B: Second input matrix (expected to be upper triangular)
            
        Returns:
            Result of A * B (which will also be upper triangular)
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=A.dtype):
            # Ensure inputs are upper triangular, although test generation should handle this.
            A_triu = torch.triu(A)
            B_triu = torch.triu(B)
            # The product of two upper triangular matrices is upper triangular.
            # No need for an extra torch.tril on the result.
            return torch.matmul(A_triu, B_triu)
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for upper triangular matrix multiplication.
        
        Returns:
            List of test case dictionaries with varying square matrix dimensions (N x N)
        """
        # Matrix dimensions: (N, N) × (N, N) = (N, N)
        test_matrices = [
            {
                "name": "2048x2048",
                "dims": (2048,),
            },
            {
                "name": "4096x4096",
                "dims": (4096,),
            },
            {
                "name": "6144x6144",
                "dims": (6144,),
            },
             {
                "name": "8192x8192",
                "dims": (8192,),
            }
        ]
        
        test_cases = []
        for matrix in test_matrices:
            seed = Problem.get_seed(f"{self.name}_{matrix['name']}")
            test_cases.append({
                "name": matrix["name"],
                "dims": matrix["dims"], # Store N as a tuple for consistency maybe? Or just N? Let's store N directly.
                "create_inputs": lambda n=matrix["dims"][0], seed=seed, dtype=dtype: (
                    *(lambda g: (
                        # Create random matrix and take upper triangle
                        torch.triu(torch.rand(n, n, device="cuda", dtype=dtype, generator=g) * 2 - 1),  # uniform [-1, 1]
                        torch.triu(torch.rand(n, n, device="cuda", dtype=dtype, generator=g) * 2 - 1)   # uniform [-1, 1]
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
        n = 4
        return {
            "name": f"{n}x{n}",
            "dims": (n,),
            "create_inputs": lambda n=n: (
                # Create upper triangular matrices for easy verification
                torch.tensor([[1.0, 2.0, 3.0, 4.0],
                            [0.0, 1.0, 2.0, 3.0],
                            [0.0, 0.0, 1.0, 2.0],
                            [0.0, 0.0, 0.0, 1.0]], device="cuda", dtype=dtype),
                torch.tensor([[1.0, 1.0, 3.0, 4.0],
                            [0.0, 8.0, 2.0, 3.0],
                            [0.0, 0.0, 1.0, 2.0],
                            [0.0, 0.0, 0.0, 1.0]], device="cuda", dtype=dtype)
            )
        }
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the upper triangular matrix multiplication result is correct.
        
        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            dtype: The data type used.
            
        Returns:
            Tuple of (is_correct, debug_info)
        """
        # Ensure the actual output is also upper triangular (optional check)
        is_upper_triangular = torch.all(torch.tril(actual_output, diagonal=-1) == 0)
        
        is_close = torch.allclose(actual_output, expected_output, rtol=1e-4, atol=1e-3) # Relaxed tolerance slightly
        
        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            max_diff = torch.max(torch.abs(diff)).item()
            mean_diff = torch.mean(torch.abs(diff)).item()
            
            debug_info = {
                "max_difference": max_diff,
                "mean_difference": mean_diff,
            }

            if not is_upper_triangular:
              debug_info["message"] = "Output is not upper triangular"
            

        return is_close, debug_info
    
    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the upper triangular matrix multiplication solution.
        
        IMPORTANT: Comments are required. Outline the FLOPs calculation.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # matrix_a (upper triangular)
                ctypes.POINTER(ctypes.c_float),  # matrix_b (upper triangular)
                ctypes.POINTER(ctypes.c_float),  # matrix_c (output, upper triangular)
                ctypes.c_size_t,                 # N (dimension of square matrices)
            ],
            "restype": None
        }
    
    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the number of floating point operations for upper triangular matrix multiplication.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            Number of floating point operations
        """
        # For C = A * B, where A, B are N x N upper triangular:
        # C[i][j] = sum_{k=0}^{j} A[i][k] * B[k][j] for i <= j
        # Number of multiply-adds for C[i][j] is j+1.
        # Total operations = sum_{i=0}^{N-1} sum_{j=0}^{i} 2*(j+1)
        # = 2 * sum_{i=0}^{N-1} ( (i+1)(i+2) / 2 )
        # = sum_{i=0}^{N-1} (i^2 + 3i + 2)
        # Let k = i+1. Sum_{k=1}^{N} ( (k-1)^2 + 3(k-1) + 2 ) = Sum (k^2 - 2k + 1 + 3k - 3 + 2) = Sum (k^2 + k)
        # = N(N+1)(2N+1)/6 + N(N+1)/2 
        # = N(N+1)/6 * (2N+1 + 3) = N(N+1)/6 * (2N+4) = N(N+1)(N+2)/3
        # Approximation: N^3 / 3 FLOPs (compared to 2*N^3 for dense)
        N = test_case["dims"][0]
        # Using the exact formula derived above
        flops = N * (N + 1) * (N + 2) // 3 
        return flops 
    
    def get_mem(self, test_case: Dict[str, Any]) -> int:
        N = test_case["dims"][0]
        
        # Input: A (N*N) + B (N*N) - both upper triangular but stored as full matrices
        # Output: C (N*N) - upper triangular but stored as full matrix
        dtype_bytes = 4  # 4 bytes per float32 element
        return (N * N + N * N + N * N) * dtype_bytes
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the dimension N
        """
        N = test_case["dims"][0]
        return [N]
