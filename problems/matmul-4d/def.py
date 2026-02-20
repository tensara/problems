import torch
from typing import List, Dict, Tuple, Any

from problem import Problem

class matmul_4d(Problem):
    """4D tensor-matrix multiplication problem."""
    
    is_exact = False

    parameters = [
        {"name": "A", "type": "float", "pointer": True, "const": True},
        {"name": "B", "type": "float", "pointer": True, "const": True},
        {"name": "C", "type": "float", "pointer": True, "const": False},
        {"name": "b", "type": "size_t", "pointer": False, "const": False},
        {"name": "i", "type": "size_t", "pointer": False, "const": False},
        {"name": "j", "type": "size_t", "pointer": False, "const": False},
        {"name": "l", "type": "size_t", "pointer": False, "const": False},
        {"name": "k", "type": "size_t", "pointer": False, "const": False},
    ]

    
    def __init__(self):
        super().__init__(
            name="matmul-4d"
        )
    
    def reference_solution(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of 4D tensor-matrix multiplication.
        
        Args:
            A: First input tensor of shape (b, i, j, l)
            B: Second input matrix of shape (l, k)
            
        Returns:
            Result of shape (b, i, j, k) from multiplying A and B along the last dimension of A
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=A.dtype):
            return torch.einsum("bijl,lk->bijk", A, B)
    
    def generate_test_cases(self) -> List[Dict[str, Any]]:
        """
        Generate test cases for 4D tensor-matrix multiplication.
        
        Returns:
            List of test case dictionaries with varying dimensions
        """
        dtype = self.param_dtype(0)

        # Tensor dimensions: (b, i, j, l) × (l, k) = (b, i, j, k)
        # dims represents (b, i, j, l, k)
        test_matrices = [
            {
                "name": "16x256x512x256 x 256x768",
                "dims": (16, 256, 512, 256, 768),
            },
            {
                "name": "8x128x256x128 x 128x512",
                "dims": (8, 128, 256, 128, 512),
            },
            {
                "name": "32x64x128x64 x 64x256",
                "dims": (32, 64, 128, 64, 256),
            },
            {
                "name": "4x32x64x32 x 32x128",
                "dims": (4, 32, 64, 32, 128),
            }
        ]
        
        test_cases = []
        for matrix in test_matrices:
            seed = Problem.get_seed(f"{self.name}_{matrix['name']}")
            test_cases.append({
                "name": matrix["name"],
                "dims": matrix["dims"],
                "create_inputs": lambda m=matrix["dims"], seed=seed, dtype=dtype: (
                    (lambda g: (
                        torch.rand(m[0], m[1], m[2], m[3], device="cuda", dtype=dtype, generator=g) * 2 - 1,  # A: (b,i,j,l)
                        torch.rand(m[3], m[4], device="cuda", dtype=dtype, generator=g) * 2 - 1         # B: (l,k)
                    ))(torch.Generator(device="cuda").manual_seed(seed))
                )
            })
        return test_cases
    
    def generate_sample(self) -> List[Dict[str, Any]]:
        """
        Generate a single sample test case for debugging or interactive runs.
        
        Returns:
            A list containing a single test case dictionary
        """
        dtype = self.param_dtype(0)

        b, i, j, l, k = (4, 4, 4, 4, 4) # Sample dimensions
        return {
            "name": f"Sample ({b}x{i}x{j}x{l} * {l}x{k})",
            "dims": (b, i, j, l, k),
            "create_inputs": lambda b=b, i=i, j=j, l=l, k_dim=k: (
                torch.arange(1, b*i*j*l + 1, device="cuda", dtype=dtype).float().view(b, i, j, l),
                torch.arange(1, l*k_dim + 1, device="cuda", dtype=dtype).float().view(l, k_dim)
            )
        }
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the tensor-matrix multiplication result is correct.
        
        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            
        Returns:
            Tuple of (is_correct, debug_info)
        """
        is_close = torch.allclose(actual_output, expected_output, rtol=2e-4, atol=6e-4)
        
        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            max_diff = torch.max(torch.abs(diff)).item()
            mean_diff = torch.mean(torch.abs(diff)).item()
            
            debug_info = {
                "max_difference": max_diff,
                "mean_difference": mean_diff
            }
        
        return is_close, debug_info
    
    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the number of floating point operations for 4D tensor-matrix multiplication.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            Number of floating point operations
        """
        # 4D tensor-matrix multiplication FLOPS = 2 * b * i * j * l * k
        # (One multiply and one add for each cell in the result, done l times, for b*i*j elements)
        b, i, j, l, k = test_case["dims"]
        return 2 * b * i * j * l * k
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the dimensions b, i, j, l, k
        """
        b, i, j, l, k = test_case["dims"]
        return [b, i, j, l, k]
