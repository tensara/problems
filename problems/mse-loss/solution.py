import torch
from typing import List, Dict, Tuple, Any
from tinygrad.tensor import Tensor

class MseLossSolutions:
    """Mixin class for Mean Squared Error loss problem solutions."""

    def reference_solution(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of MSE loss function.

        Args:
            predictions: Predicted values tensor of arbitrary shape
            targets: Target values tensor of the same shape as predictions

        Returns:
            Mean squared error loss as a scalar tensor
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            return torch.mean((predictions - targets) ** 2)

    def reference_tinygrad_solution(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Tinygrad implementation of MSE loss function.

        Args:
            predictions: Predicted values tensor of arbitrary shape
            targets: Target values tensor of the same shape as predictions

        Returns:
            Mean squared error loss as a scalar tensor
        """
        # MSE = mean((predictions - targets)^2)
        return (predictions - targets).pow(2).mean()
