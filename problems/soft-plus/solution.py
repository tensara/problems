import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any
from tinygrad.tensor import Tensor

class SoftPlusSolutions:
    """Mixin class for Softplus activation function problem solutions."""

    def reference_solution(self, input_matrix: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of Softplus.

        Args:
            input_matrix: Input matrix of shape (M, N)

        Returns:
            Result of Softplus activation
        """
        with torch.no_grad():
            return F.softplus(input_matrix)

    def reference_tinygrad_solution(self, input_matrix: Tensor) -> Tensor:
        """
        Tinygrad implementation of Softplus.

        Args:
            input_matrix: Input matrix of shape (M, N)

        Returns:
            Result of Softplus activation
        """
        # Softplus: log(1 + exp(x))
        return (1 + input_matrix.exp()).log()
