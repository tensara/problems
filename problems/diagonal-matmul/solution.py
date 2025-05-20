import torch
from typing import List, Dict, Tuple, Any

class DiagonalMatmulSolutions:
    """Mixin class for diagonal matrix multiplication problem solutions."""

    def reference_solution(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of diagonal matrix multiplication.

        Args:
            A: 1D tensor representing the diagonal of the diagonal matrix
            B: 2D tensor representing the second matrix

        Returns:
            Result of diag(A) * B
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            return torch.diag(A) @ B
