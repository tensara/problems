import torch
from typing import List, Dict, Tuple, Any
from tinygrad.tensor import Tensor

class L2NormSolutions:
    """Mixin class for L2 Normalization problem solutions."""

    epsilon = 1e-10  # Small epsilon for numerical stability

    def reference_solution(self, x: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of L2 Normalization.

        Args:
            x (torch.Tensor): Input tensor of shape (B, D)

        Returns:
            torch.Tensor: Output tensor with L2 Normalization applied, same shape as input.
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            l2_norm = torch.norm(x, p=2, dim=1, keepdim=True)

            l2_norm = l2_norm + self.epsilon

            output = x / l2_norm

            return output

    def reference_tinygrad_solution(self, x: Tensor) -> Tensor:
        """
        Tinygrad implementation of L2 Normalization.

        Args:
            x (Tensor): Input tensor of shape (B, D)

        Returns:
            Tensor: Output tensor with L2 Normalization applied, same shape as input.
        """
        l2_norm = x.pow(2).sum(axis=1, keepdim=True).sqrt()

        l2_norm = l2_norm + self.epsilon

        output = x / l2_norm

        return output
