import torch
from typing import List, Dict, Tuple, Any
from tinygrad.tensor import Tensor

class TanhSolutions:
    """Mixin class for Tanh activation function problem solutions."""

    def reference_solution(self, input_matrix: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of Tanh.

        Args:
            input_matrix: Input matrix of shape (M, N)

        Returns:
            Result of Tanh activation
        """
        with torch.no_grad():
            return torch.tanh(input_matrix)

    def reference_tinygrad_solution(self, input_matrix: Tensor) -> Tensor:
        """
        Tinygrad implementation of Tanh.

        Args:
            input_matrix: Input matrix of shape (M, N)

        Returns:
            Result of Tanh activation
        """
        # Tinygrad's tanh is the direct equivalent of PyTorch's torch.tanh
        return input_matrix.tanh()
