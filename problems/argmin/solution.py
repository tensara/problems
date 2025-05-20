import torch
from tinygrad.tensor import Tensor
from typing import List, Dict, Tuple, Any

class ArgminSolutions:
    """Mixin class for argmin problem solutions."""

    def reference_solution(self, input_tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """
        PyTorch implementation of argmin over dimension.

        Args:
            input_tensor: Input tensor of arbitrary shape
            dim: Dimension to perform argmin over

        Returns:
            Result of argmin operation
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            return torch.argmin(input_tensor, dim=dim).to(torch.int32)

    def reference_tinygrad_solution(self, input_tensor: Tensor, dim: int) -> Tensor:
        """
        Tinygrad implementation of argmin over dimension.

        Args:
            input_tensor: Input tensor of arbitrary shape
            dim: Dimension to perform argmin over

        Returns:
            Result of argmin operation
        """
        # Tinygrad's argmin is the direct equivalent of PyTorch's torch.argmin
        return input_tensor.argmin(axis=dim)
