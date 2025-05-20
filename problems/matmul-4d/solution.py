import torch
from typing import List, Dict, Tuple, Any

class Matmul4dSolutions:
    """Mixin class for 4D tensor-matrix multiplication problem solutions."""

    def reference_solution(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of 4D tensor-matrix multiplication.

        Args:
            A: First input tensor of shape (b, i, j, l)
            B: Second input matrix of shape (l, k)

        Returns:
            Result of shape (b, i, j, k) from multiplying A and B along the last dimension of A
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            return torch.einsum("bijl,lk->bijk", A, B)
