import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any
from tinygrad.tensor import Tensor

class EluSolutions:
    """Mixin class for ELU (Exponential Linear Unit) activation function problem solutions."""

    alpha = 1.0  # Default alpha value for ELU

    def reference_solution(self, input_matrix: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of ELU.

        Args:
            input_matrix: Input matrix of shape (M, N)

        Returns:
            Result of ELU activation
        """
        with torch.no_grad():
            return F.elu(input_matrix, alpha=self.alpha)

    def reference_tinygrad_solution(self, input_matrix: Tensor) -> Tensor:
        """
        Tinygrad implementation of ELU.

        Args:
            input_matrix: Input matrix of shape (M, N)

        Returns:
            Result of ELU activation
        """
        # ELU(x) = x if x > 0, else alpha * (exp(x) - 1)
        return input_matrix.relu() + self.alpha * (input_matrix.exp() - 1) * (1 - input_matrix.relu().sign())
