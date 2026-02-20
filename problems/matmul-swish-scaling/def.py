import torch
from typing import List, Dict, Tuple, Any

from problem import Problem

class matmul_swish_scaling(Problem):
    """Matrix multiplication followed by Swish activation followed by scaling fusion problem."""
    
    is_exact = False

    parameters = [
        {"name": "A", "type": "float", "pointer": True, "const": True},
        {"name": "B", "type": "float", "pointer": True, "const": True},
        {"name": "scale", "type": "float", "pointer": False, "const": False},
        {"name": "output", "type": "float", "pointer": True, "const": False},
        {"name": "M", "type": "size_t", "pointer": False, "const": False},
        {"name": "N", "type": "size_t", "pointer": False, "const": False},
        {"name": "K", "type": "size_t", "pointer": False, "const": False},
    ]

    
    def __init__(self):
        super().__init__(
            name="matmul-swish-scaling"
        )
    
    def reference_solution(self, A: torch.Tensor, B: torch.Tensor, scale: float) -> torch.Tensor:
        """
        PyTorch implementation of matrix multiplication followed by Swish followed by scaling.
        
        Args:
            A: First input matrix
            B: Second input matrix
            scale: Scaling factor
            
        Returns:
            Result of scale * swish(A * B)
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=A.dtype):
            matmul_result = torch.matmul(A, B)
            swish_result = matmul_result * torch.sigmoid(matmul_result)
            scaled_result = scale * swish_result
            return scaled_result
    
    def generate_test_cases(self) -> List[Dict[str, Any]]:
        """
        Generate test cases for matmul-swish-scaling fusion.
        
        Returns:
            List of test case dictionaries with varying matrix dimensions
        """
        dtype = self.param_dtype(0)

        # Matrix dimensions: (M, K) × (K, N) = (M, N)
        # dims represents (M, N, K)
        test_matrices = [
            {
                "name": "512x512 x 512x512",
                "dims": (512, 512, 512),
                "scale": 0.5,
            },
            {
                "name": "1024x1024 x 1024x512", 
                "dims": (1024, 512, 1024),
                "scale": 1.414,
            },
            {
                "name": "512x512 x 512x1024",
                "dims": (512, 1024, 512),
                "scale": 2.0,
            },
            {
                "name": "1024x1024 x 1024x1024",
                "dims": (1024, 1024, 1024),
                "scale": 0.707,
            }
        ]
        
        test_cases = []
        for matrix in test_matrices:
            seed = Problem.get_seed(f"{self.name}_{matrix['name']}")
            test_cases.append({
                "name": matrix["name"],
                "dims": matrix["dims"],
                "scale": matrix["scale"],
                "create_inputs": lambda m=matrix["dims"], s=matrix["scale"], seed=seed, dtype=dtype: (
                    *(lambda g: (
                        torch.rand(m[0], m[2], device="cuda", dtype=dtype, generator=g) * 2 - 1,  # uniform [-1, 1]
                        torch.rand(m[2], m[1], device="cuda", dtype=dtype, generator=g) * 2 - 1,  # uniform [-1, 1]
                    ))(torch.Generator(device="cuda").manual_seed(seed)),
                    s
                )
            })
        return test_cases
    
    def generate_sample(self) -> Dict[str, Any]:
        """
        Generate sample test case for matmul-swish-scaling with predictable inputs.

        Returns:
            Dictionary containing the sample test case.
        """
        dtype = self.param_dtype(0)

        m_dims = (4, 4, 4)  # M, N, K dimensions
        scale = 1.5
        return {
            "name": "4x4_square",
            "dims": m_dims,
            "scale": scale,
            "create_inputs": lambda m_dims=m_dims, scale=scale: (
                torch.tensor([
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0]
                ], device="cuda", dtype=dtype),
                torch.tensor([
                    [1.0, -2.0, 3.0, -4.0],
                    [-5.0, 6.0, -7.0, 8.0],
                    [9.0, -10.0, 11.0, -12.0],
                    [-13.0, 14.0, -15.0, 16.0]
                ], device="cuda", dtype=dtype),
                scale
            )
        }
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the matmul-swish-scaling result is correct.
        
        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            
        Returns:
            Tuple of (is_correct, debug_info)
        """
        is_close = torch.allclose(actual_output, expected_output, rtol=5e-4, atol=2e-4)
        
        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            max_diff = torch.max(torch.abs(diff)).item()
            mean_diff = torch.mean(torch.abs(diff)).item()
            
            # Find indices of largest differences
            flat_diff = diff.flatten()
            _, top_indices = torch.topk(torch.abs(flat_diff), min(5, flat_diff.numel()))
            
            # Convert flat indices back to 2D coordinates
            m, n = expected_output.shape
            sample_diffs = {}
            for i, idx in enumerate(top_indices):
                row = idx.item() // n
                col = idx.item() % n
                sample_diffs[f"({row}, {col})"] = {
                    "expected": expected_output[row, col].item(),
                    "actual": actual_output[row, col].item(),
                    "diff": diff[row, col].item()
                }
            
            debug_info = {
                "max_difference": max_diff,
                "mean_difference": mean_diff,
                "sample_differences": sample_diffs
            }
        
        return is_close, debug_info
    
    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the number of floating point operations for the problem.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            Number of floating point operations
        """
        # Matrix multiplication FLOPS = 2 * M * N * K
        # Swish FLOPS = 4 * M * N (sigmoid computation + multiplication)
        # Scaling FLOPS = M * N (one multiplication per element)
        M, N, K = test_case["dims"]
        matmul_flops = 2 * M * N * K
        swish_flops = 4 * M * N
        scaling_flops = M * N
        return matmul_flops + swish_flops + scaling_flops
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the scale factor and dimensions M, N, K
        """
        M, N, K = test_case["dims"]
        return [M, N, K] 