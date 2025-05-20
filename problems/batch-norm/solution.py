import torch
import torch.nn as nn
from tinygrad.tensor import Tensor
from typing import List, Dict, Tuple, Any

class BatchNormSolutions:
    """Mixin class for Batch Normalization problem solutions."""

    epsilon = 1e-5  # Standard epsilon for BatchNorm

    def reference_solution(self, x: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of Batch Normalization using nn.BatchNorm2d.

        Args:
            x (torch.Tensor): Input tensor of shape (B, F, D1, D2)

        Returns:
            torch.Tensor: Output tensor with Batch Normalization applied, same shape as input.
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            # Create BatchNorm2d layer with no affine parameters and no running stats
            bn = nn.BatchNorm2d(
                num_features=x.size(1),  # F dimension
                affine=False,  # No learnable parameters
                track_running_stats=False,  # Don't track running stats
                eps=self.epsilon
            )
            return bn(x)

    def reference_tinygrad_solution(self, x: Tensor) -> Tensor:
        """
        Tinygrad implementation of Batch Normalization (without affine or running stats).

        Args:
            x (Tensor): Input tensor of shape (B, F, D1, D2)

        Returns:
            Tensor: Output tensor with Batch Normalization applied, same shape as input.
        """
        # Calculate mean over batch and spatial dimensions (0, 2, 3)
        mean = x.mean(axis=(0, 2, 3), keepdim=True)

        # Calculate variance over batch and spatial dimensions (0, 2, 3)
        # Variance = mean((x - mean)^2)
        variance = ((x - mean)**2).mean(axis=(0, 2, 3), keepdim=True)

        # Normalize
        # output = (x - mean) / sqrt(variance + epsilon)
        output = (x - mean) / (variance + self.epsilon).sqrt()

        return output
