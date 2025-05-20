import torch
from typing import List, Dict, Tuple, Any

class MaxDimSolutions:
    """Mixin class for Max over dimension problem solutions."""

    def reference_solution(self, input_tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """
        PyTorch implementation of max over dimension.

        Args:
            input_tensor: Input tensor of arbitrary shape
            dim: Dimension to reduce over

        Returns:
            Result of max reduction with keepdim=True (values only)
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            # Get only the values, not the indices
            return torch.max(input_tensor, dim=dim, keepdim=True)[0]
