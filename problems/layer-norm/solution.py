import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any
from tinygrad.tensor import Tensor

class LayerNormSolutions:
    """Mixin class for Layer Normalization problem solutions."""

    epsilon = 1e-5 # Standard epsilon for LayerNorm

    def reference_solution(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of Layer Normalization.

        Args:
            x (torch.Tensor): Input tensor of shape (B, F, D1, D2)
            gamma (torch.Tensor): Scale tensor of shape (F, D1, D2)
            beta (torch.Tensor): Shift tensor of shape (F, D1, D2)

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            # Normalize over the last 3 dimensions (F, D1, D2)
            normalized_shape = x.shape[1:]

            # Use torch.nn.functional.layer_norm
            output = F.layer_norm(
                x,
                normalized_shape,
                weight=gamma,
                bias=beta,
                eps=self.epsilon
            )
            return output

    def reference_tinygrad_solution(self, x: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
        """
        Tinygrad implementation of Layer Normalization.

        Args:
            x (Tensor): Input tensor of shape (B, F, D1, D2)
            gamma (Tensor): Scale tensor of shape (F, D1, D2)
            beta (Tensor): Shift tensor of shape (F, D1, D2)

        Returns:
            Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        # Normalize over the last 3 dimensions (F, D1, D2)
        # Calculate mean and variance over the last 3 dimensions
        mean = x.mean(axis=(1, 2, 3), keepdim=True)
        variance = ((x - mean)**2).mean(axis=(1, 2, 3), keepdim=True)

        # Normalize
        output = (x - mean) / (variance + self.epsilon).sqrt()

        # Apply scale and shift
        output = output * gamma + beta

        return output
