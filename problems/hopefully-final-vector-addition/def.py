import torch
from tensara.problem import Problem

class MyProblem(Problem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_function_signature(self):
        return ("a", "b",)

    def reference_solution(self, a, b):
        import torch
        
        def vector_addition(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
          """
          Computes the element-wise sum of two tensors.
          """
          return a + b

    def verify_result(self, my_result, torch_result):
        return torch.allclose(my_result, torch_result)

    def generate_sample(self):
        # This should be implemented by the user to generate a sample input
        pass

    def generate_test_cases(self):
        return [
            {"input": """[1, 2, 3, 4, 5, 6, 7, 8]
[8, 7, 6, 5, 4, 3, 2, 1]""", "output": """[9, 9, 9, 9, 9, 9, 9, 9]"""},
            {"input": """[10.5, -2.0, 0.0, 100.1]
[-5.5, 2.0, 1.0, -0.1]""", "output": """[5.0, 0.0, 1.0, 100.0]"""},
            {"input": """[0]
[0]""", "output": """[0]"""},
        ]

    def get_flops(self, a, b):
        return N

    def get_extra_params(self):
        return {}
