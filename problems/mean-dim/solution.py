import torch
from typing import List, Dict, Tuple, Any
from tinygrad.tensor import Tensor

class MeanDimSolutions:
    """Mixin class for Mean over dimension problem solutions."""

    def reference_solution(self, input_tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """
        PyTorch implementation of mean over dimension.

        Args:
            input_tensor: Input tensor of arbitrary shape
            dim: Dimension to reduce over

        Returns:
            Result of mean reduction with keepdim=True
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            return torch.mean(input_tensor, dim=dim, keepdim=True)

    def reference_tinygrad_solution(self, input_tensor: Tensor, dim: int) -> Tensor:
        """
        Tinygrad implementation of mean over dimension.

        Args:
            input_tensor: Input tensor of arbitrary shape
            dim: Dimension to reduce over

        Returns:
            Result of mean reduction with keepdim=True
        """
        # Tinygrad's mean supports specifying the axis and keeping the dimension
        return input_tensor.mean(axis=dim, keepdim=True)
