import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any
from tinygrad.tensor import Tensor

class HuberLossSolutions:
    """Mixin class for Huber Loss (Smooth L1 Loss) problem solutions."""

    def reference_solution(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of element-wise Huber Loss (Smooth L1 Loss).

        Args:
            predictions: Predictions tensor of shape (N,)
            targets: Targets tensor of shape (N,)

        Returns:
            Element-wise Huber loss tensor of shape (N,)
        """
        with torch.no_grad():
            # Use reduction='none' to get element-wise loss
            return F.smooth_l1_loss(predictions, targets, reduction='none', beta=1.0)

    def reference_tinygrad_solution(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Tinygrad implementation of element-wise Huber Loss (Smooth L1 Loss).

        Args:
            predictions: Predictions tensor of shape (N,)
            targets: Targets tensor of shape (N,)

        Returns:
            Element-wise Huber loss tensor of shape (N,)
        """
        # Huber Loss (Smooth L1 Loss) with delta = 1.0
        # f(x) = 0.5 * x^2 if |x| < 1
        # f(x) = |x| - 0.5 otherwise
        x = predictions - targets
        abs_x = x.abs()
        square_loss = 0.5 * x.pow(2)
        linear_loss = abs_x - 0.5
        return square_loss * (abs_x < 1.0) + linear_loss * (abs_x >= 1.0)
