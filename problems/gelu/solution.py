import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any
from tinygrad.tensor import Tensor
import math

class GeluSolutions:
    """Mixin class for GELU activation function problem solutions."""

    def reference_solution(self, input_matrix: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of GELU.

        Args:
            input_matrix: Input matrix of shape (M, N)

        Returns:
            Result of GELU activation
        """
        with torch.no_grad():
            return F.gelu(input_matrix)

    def reference_tinygrad_solution(self, input_matrix: Tensor) -> Tensor:
        """
        Tinygrad implementation of GELU (approximate).

        Args:
            input_matrix: Input matrix of shape (M, N)

        Returns:
            Result of GELU activation
        """
        # Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x.pow(3))))
        x = input_matrix
        return 0.5 * x * (1 + (math.sqrt(2 / math.pi) * (x + 0.044715 * x.pow(3))).tanh())
