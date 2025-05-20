import torch
from typing import List, Dict, Tuple, Any
from tinygrad.tensor import Tensor
import math

class SeluSolutions:
    """Mixin class for SELU (Scaled Exponential Linear Unit) activation function problem solutions."""

    # Predefined constants for SELU
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    def reference_solution(self, input_matrix: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of SELU.

        Args:
            input_matrix: Input matrix of shape (M, N)

        Returns:
            Result of SELU activation
        """
        with torch.no_grad():
            return torch.selu(input_matrix)

    def reference_tinygrad_solution(self, input_matrix: Tensor) -> Tensor:
        """
        Tinygrad implementation of SELU.

        Args:
            input_matrix: Input matrix of shape (M, N)

        Returns:
            Result of SELU activation
        """
        # SELU(x) = scale * x if x > 0, else scale * alpha * (exp(x) - 1)
        x = input_matrix
        return self.scale * (x.relu() + self.alpha * (x.exp() - 1) * (1 - x.relu().sign()))
