import torch
from tinygrad.tensor import Tensor
from typing import List, Dict, Tuple, Any

class ArgmaxSolutions:
    """Mixin class for argmax problem solutions."""

    def reference_solution(self, input_tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """
        PyTorch implementation of argmax over dimension.

        Args:
            input_tensor: Input tensor of arbitrary shape
            dim: Dimension to perform argmax over

        Returns:
            Result of argmax operation
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            return torch.argmax(input_tensor, dim=dim).to(torch.int32)

    def reference_tinygrad_solution(self, input_tensor: Tensor, dim: int) -> Tensor:
        """
        Tinygrad implementation of argmax over dimension.

        Args:
            input_tensor: Input tensor of arbitrary shape
            dim: Dimension to perform argmax over

        Returns:
            Result of argmax operation
        """
        # Tinygrad's argmax is the direct equivalent of PyTorch's torch.argmax
        return input_tensor.argmax(axis=dim)
