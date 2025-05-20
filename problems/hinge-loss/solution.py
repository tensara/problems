import torch
from typing import List, Dict, Tuple, Any
from tinygrad.tensor import Tensor

class HingeLossSolutions:
    """Mixin class for Hinge Loss problem solutions."""

    def reference_solution(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of element-wise Hinge Loss.

        Args:
            predictions: Predictions tensor of shape (N,)
            targets: Binary targets tensor of shape (N,) with values in {-1, 1}

        Returns:
            Element-wise hinge loss tensor of shape (N,)
        """
        with torch.no_grad():
            return torch.clamp(1 - predictions * targets, min=0)

    def reference_tinygrad_solution(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Tinygrad implementation of element-wise Hinge Loss.

        Args:
            predictions: Predictions tensor of shape (N,)
            targets: Binary targets tensor of shape (N,) with values in {-1, 1}

        Returns:
            Element-wise hinge loss tensor of shape (N,)
        """
        # Hinge Loss: max(0, 1 - predictions * targets)
        return (1 - predictions * targets).relu()
