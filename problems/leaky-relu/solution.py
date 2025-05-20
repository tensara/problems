import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any
from tinygrad.tensor import Tensor

class LeakyReluSolutions:
    """Mixin class for Leaky ReLU activation function problem solutions."""

    def reference_solution(self, input_matrix: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        PyTorch implementation of Leaky ReLU.

        Args:
            input_matrix: Input matrix of shape (M, N)
            alpha: Slope for negative values

        Returns:
            Result of Leaky ReLU activation
        """
        with torch.no_grad():
            return F.leaky_relu(input_matrix, alpha)

    def reference_tinygrad_solution(self, input_matrix: Tensor, alpha: float) -> Tensor:
        """
        Tinygrad implementation of Leaky ReLU.

        Args:
            input_matrix: Input matrix of shape (M, N)
            alpha: Slope for negative values

        Returns:
            Result of Leaky ReLU activation
        """
        # Leaky ReLU: x if x > 0, else alpha * x
        return input_matrix.relu() - (-input_matrix).relu() * (-alpha)
