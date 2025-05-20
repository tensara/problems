import torch
from typing import List, Dict, Tuple, Any
from tinygrad.tensor import Tensor

class SwishSolutions:
    """Mixin class for Swish activation function problem solutions."""

    def reference_solution(self, input_matrix: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of Swish.

        Args:
            input_matrix: Input matrix of shape (M, N)

        Returns:
            Result of Swish activation
        """
        with torch.no_grad():
            return input_matrix  * torch.sigmoid(input_matrix)

    def reference_tinygrad_solution(self, input_matrix: Tensor) -> Tensor:
        """
        Tinygrad implementation of Swish.

        Args:
            input_matrix: Input matrix of shape (M, N)

        Returns:
            Result of Swish activation
        """
        # Swish: x * sigmoid(x)
        return input_matrix * input_matrix.sigmoid()
