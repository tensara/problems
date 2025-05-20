import torch
from typing import List, Dict, Tuple, Any

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
