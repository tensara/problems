import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any

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
