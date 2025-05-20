import torch
from typing import List, Dict, Tuple, Any

class LowerTrigMatmulSolutions:
    """Mixin class for lower triangular matrix multiplication problem solutions."""

    def reference_solution(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of lower triangular matrix multiplication.
        Ensures inputs are lower triangular before multiplying.

        Args:
            A: First input matrix (expected to be lower triangular)
            B: Second input matrix (expected to be lower triangular)

        Returns:
            Result of A * B (which will also be lower triangular)
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            # Ensure inputs are lower triangular, although test generation should handle this.
            A_tril = torch.tril(A)
            B_tril = torch.tril(B)
            # The product of two lower triangular matrices is lower triangular.
            # No need for an extra torch.tril on the result.
            return torch.matmul(A_tril, B_tril)
