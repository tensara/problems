import torch
from typing import List, Dict, Tuple, Any
from tinygrad.tensor import Tensor

class CumsumSolutions:
    """Mixin class for cumulative sum problem solutions."""

    def reference_solution(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of cumulative sum.

        Args:
            input_tensor: Input tensor of shape (N)

        Returns:
            Cumulative sum of the input tensor
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            return torch.cumsum(input_tensor, dim=0)

    def reference_tinygrad_solution(self, input_tensor: Tensor) -> Tensor:
        """
        Tinygrad implementation of cumulative sum.

        Args:
            input_tensor: Input tensor of shape (N)

        Returns:
            Cumulative sum of the input tensor
        """
        # Tinygrad's cumsum is the direct equivalent of PyTorch's torch.cumsum
        return input_tensor.cumsum(axis=0)
