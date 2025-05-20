import torch
from typing import List, Dict, Tuple, Any
from tinygrad.tensor import Tensor

class ReluSolutions:
    """Mixin class for ReLU activation function problem solutions."""

    def reference_solution(self, input_matrix: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of ReLU.

        Args:
            input_matrix: Input matrix of shape (M, N)

        Returns:
            Result of ReLU activation
        """
        with torch.no_grad():
            return torch.relu(input_matrix)

    def reference_tinygrad_solution(self, input_matrix: Tensor) -> Tensor:
        """
        Tinygrad implementation of ReLU.

        Args:
            input_matrix: Input matrix of shape (M, N)

        Returns:
            Result of ReLU activation
        """
        # Tinygrad's relu is the direct equivalent of PyTorch's torch.relu
        return input_matrix.relu()
