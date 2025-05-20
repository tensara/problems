import torch
from typing import List, Dict, Tuple, Any
from tinygrad.tensor import Tensor

class UpperTrigMatmulSolutions:
    """Mixin class for upper triangular matrix multiplication problem solutions."""

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
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            # Ensure inputs are upper triangular, although test generation should handle this.
            A_triu = torch.triu(A)
            B_triu = torch.triu(B)
            # The product of two upper triangular matrices is upper triangular.
            # No need for an extra torch.tril on the result.
            return torch.matmul(A_triu, B_triu)

    def reference_tinygrad_solution(self, A: Tensor, B: Tensor) -> Tensor:
        """
        Tinygrad implementation of upper triangular matrix multiplication.
        Ensures inputs are upper triangular before multiplying.

        Args:
            A: First input matrix (expected to be upper triangular)
            B: Second input matrix (expected to be upper triangular)

        Returns:
            Result of A * B (which will also be upper triangular)
        """
        # Ensure inputs are upper triangular
        A_triu = A.triu()
        B_triu = B.triu()
        # The product of two upper triangular matrices is upper triangular.
        # No need for an extra triu on the result.
        return A_triu @ B_triu
