import torch
from typing import List, Dict, Tuple, Any

class MatrixVectorSolutions:
    """Mixin class for matrix vector multiplication problem solutions."""

    def reference_solution(self, matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of matrix-vector multiplication.

        Args:
            matrix: Input matrix of shape (M, K)
            vector: Input vector of shape (K)

        Returns:
            Result of matrix-vector multiplication of shape (M)
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            return torch.matmul(matrix, vector)
