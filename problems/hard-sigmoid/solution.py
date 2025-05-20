import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any
from tinygrad.tensor import Tensor

class HardSigmoidSolutions:
    """Mixin class for Hard Sigmoid activation function problem solutions."""

    def reference_solution(self, input_matrix: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of Hard Sigmoid.

        Args:
            input_matrix: Input matrix of shape (M, N)

        Returns:
            Result of Hard Sigmoid activation
        """
        with torch.no_grad():
            return F.hardsigmoid(input_matrix)

    def reference_tinygrad_solution(self, input_matrix: Tensor) -> Tensor:
        """
        Tinygrad implementation of Hard Sigmoid.

        Args:
            input_matrix: Input matrix of shape (M, N)

        Returns:
            Result of Hard Sigmoid activation
        """
        # Hard Sigmoid: max(0, min(1, x * 0.16666 + 0.5))
        return (input_matrix * (1.0/6.0) + 0.5).clip(0, 1)
