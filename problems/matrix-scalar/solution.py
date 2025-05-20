import torch
from typing import List, Dict, Tuple, Any
from tinygrad.tensor import Tensor

class MatrixScalarSolutions:
    """Mixin class for matrix scalar multiplication problem solutions."""

    def reference_solution(self, matrix_a: torch.Tensor, scalar: float) -> torch.Tensor:
        """
        PyTorch implementation of matrix scalar multiplication.

        Args:
            matrix_a: First input matrix of shape (N, N)
            scalar: Second input scalar

        Returns:
            Result of matrix scalar multiplication of shape (N, N)
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            return matrix_a * scalar

    def reference_tinygrad_solution(self, matrix_a: Tensor, scalar: float) -> Tensor:
        """
        Tinygrad implementation of matrix scalar multiplication.

        Args:
            matrix_a: First input matrix of shape (N, N)
            scalar: Second input scalar

        Returns:
            Result of matrix scalar multiplication of shape (N, N)
        """
        # Tinygrad's * operator performs scalar multiplication
        return matrix_a * scalar
