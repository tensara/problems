import torch
from typing import List, Dict, Tuple, Any

class L1NormSolutions:
    """Mixin class for L1 Normalization problem solutions."""

    epsilon = 1e-10  # Small epsilon for numerical stability

    def reference_solution(self, x: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of L1 Normalization.

        Args:
            x (torch.Tensor): Input tensor of shape (B, D)

        Returns:
            torch.Tensor: Output tensor with L1 Normalization applied, same shape as input.
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            l1_norm = torch.sum(torch.abs(x), dim=1, keepdim=True)

            l1_norm = l1_norm + self.epsilon

            output = x / l1_norm

            return output
