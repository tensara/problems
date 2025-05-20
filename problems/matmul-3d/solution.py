import torch
from typing import List, Dict, Tuple, Any
from tinygrad.tensor import Tensor

class Matmul3dSolutions:
    """Mixin class for 3D matrix multiplication problem solutions."""

    def reference_solution(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of 3D tensor-matrix multiplication.

        Args:
            A: First input tensor of shape (N, M, K)
            B: Second input matrix of shape (K, L)

        Returns:
            Result of shape (N, M, L) from multiplying A and B along the last dimension of A
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            return torch.matmul(A, B)

    def reference_tinygrad_solution(self, A: Tensor, B: Tensor) -> Tensor:
        """
        Tinygrad implementation of 3D tensor-matrix multiplication.

        Args:
            A: First input tensor of shape (N, M, K)
            B: Second input matrix of shape (K, L)

        Returns:
            Result of shape (N, M, L) from multiplying A and B along the last dimension of A
        """
        # Tinygrad's @ operator handles broadcasting for 3D tensor-matrix multiplication
        return A @ B
