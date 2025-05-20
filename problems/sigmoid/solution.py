import torch
from typing import List, Dict, Tuple, Any
from tinygrad.tensor import Tensor

class SigmoidSolutions:
    """Mixin class for Sigmoid activation function problem solutions."""

    def reference_solution(self, input_matrix: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of Sigmoid.

        Args:
            input_matrix: Input matrix of shape (M, N)

        Returns:
            Result of Sigmoid activation
        """
        with torch.no_grad():
            return torch.sigmoid(input_matrix)

    def reference_tinygrad_solution(self, input_matrix: Tensor) -> Tensor:
        """
        Tinygrad implementation of Sigmoid.

        Args:
            input_matrix: Input matrix of shape (M, N)

        Returns:
            Result of Sigmoid activation
        """
        # Sigmoid: 1 / (1 + exp(-x))
        return 1 / (1 + (-input_matrix).exp())
