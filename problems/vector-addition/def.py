import torch
from typing import List, Dict, Tuple, Any

from problem import Problem

class vector_addition(Problem):
    """Vector addition problem."""
    
    is_exact = False

    parameters = [
        {"name": "d_input1", "type": "float", "pointer": True, "const": True},
        {"name": "d_input2", "type": "float", "pointer": True, "const": True},
        {"name": "d_output", "type": "float", "pointer": True, "const": False},
        {"name": "n", "type": "size_t", "pointer": False, "const": False},
    ]

    
    def __init__(self):
        super().__init__(
            name="vector-addition"
        )
    
    def reference_solution(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of vector addition.
        
        Args:
            A: First input tensor
            B: Second input tensor
            
        Returns:
            Result of A + B
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=A.dtype):
            return A + B
    
    def generate_test_cases(self) -> List[Dict[str, Any]]:
        """
        Generate test cases for vector addition.
        
        Returns:
            List of test case dictionaries with varying sizes
        """
        dtype = self.param_dtype(0)

        sizes = [
            ("n = 2^20", 1048576),
            ("n = 2^22", 4194304),
            ("n = 2^23", 8388608),
            ("n = 2^25", 33554432),
            ("n = 2^26", 67108864),
            ("n = 2^29", 536870912),
            ("n = 2^30", 1073741824),
        ]
        
        test_cases = []
        for name, size in sizes:
            seed = Problem.get_seed(f"{self.name}_{name}_{(size,)}")
            test_cases.append({
                "name": name,
                "dims": (size,),
                "create_inputs": lambda size=size, seed=seed, dtype=dtype: (
                    *(lambda g: (
                        torch.rand(size, device="cuda", dtype=dtype, generator=g) * 2 - 1,  # uniform [-1, 1]
                        torch.rand(size, device="cuda", dtype=dtype, generator=g) * 2 - 1   # uniform [-1, 1]
                    ))(torch.Generator(device="cuda").manual_seed(seed)),
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

        name = "Sample (n = 8)"
        size = 8 
        return {
            "name": name,
            "dims": (size,),
            "create_inputs": lambda size=size: ( # Ensure size is captured
                torch.tensor([1.0, 2.0, 3.0, 4.0, -1.0, -2.0, 0.0, 0.5], device="cuda", dtype=dtype),
                torch.tensor([0.5, -1.0, 1.0, 0.0, 1.0, -1.0, 0.5, 2.0], device="cuda", dtype=dtype),
            )
        }
 
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the vector addition result is correct.
        
        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            
        Returns:
            Tuple of (is_correct, debug_info)
        """
        is_close = torch.allclose(actual_output, expected_output, rtol=2e-4, atol=1e-4)
        
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
        Get the number of floating point operations for the problem.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            Number of floating point operations
        """
        # Vector addition has 1 FLOP per element
        N = test_case["dims"][0]
        return N
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the vector length N
        """
        N = test_case["dims"][0]
        return [N]