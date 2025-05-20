import torch
from typing import List, Dict, Tuple, Any

class CumprodSolutions:
    """Mixin class for cumulative product problem solutions."""

    def reference_solution(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of cumulative product.

        Args:
            input_tensor: Input tensor of shape (N)

        Returns:
            Cumulative product of the input tensor
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            return torch.cumprod(input_tensor, dim=0)
