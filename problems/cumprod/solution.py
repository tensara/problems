import torch
from typing import List, Dict, Tuple, Any
from tinygrad.tensor import Tensor

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

    def reference_tinygrad_solution(self, input_tensor: Tensor) -> Tensor:
        """
        Tinygrad implementation of cumulative product.

        Args:
            input_tensor: Input tensor of shape (N)

        Returns:
            Cumulative product of the input tensor
        """
        # Tinygrad's cumprod is the direct equivalent of PyTorch's torch.cumprod
        return input_tensor.cumprod(axis=0)
