import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any

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
