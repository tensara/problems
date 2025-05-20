import torch
from typing import List, Dict, Tuple, Any
from tinygrad.tensor import Tensor

class SquareMatmulSolutions:
    """Mixin class for square matrix multiplication problem solutions."""

    def reference_solution(self, matrix_a: torch.Tensor, matrix_b: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of square matrix multiplication.

        Args:
            matrix_a: First input matrix of shape (N, N)
            matrix_b: Second input matrix of shape (N, N)

        Returns:
            Result of matrix multiplication of shape (N, N)
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            return torch.matmul(matrix_a, matrix_b)

    def reference_tinygrad_solution(self, matrix_a: Tensor, matrix_b: Tensor) -> Tensor:
        """
        Tinygrad implementation of square matrix multiplication.

        Args:
            matrix_a: First input matrix of shape (N, N)
            matrix_b: Second input matrix of shape (N, N)

        Returns:
            Result of matrix multiplication of shape (N, N)
        """
        # Tinygrad's @ operator performs matrix multiplication
        return matrix_a @ matrix_b
