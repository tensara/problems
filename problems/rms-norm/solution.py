import torch
from typing import List, Dict, Tuple, Any

class RmsNormSolutions:
    """Mixin class for RMS Normalization problem solutions."""

    epsilon = 1e-5  # Standard epsilon for RMSNorm

    def reference_solution(self, x: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of RMS Normalization.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            # Calculate the RMS along the feature dimension
            # For 2D inputs (batch_size, num_features), this is along dim=1
            # For higher dimensional inputs, still use dim=1 (the feature dimension)
            rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

            # Normalize the input by dividing by the RMS
            output = x / rms

            return output
