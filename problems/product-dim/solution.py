import torch
from typing import List, Dict, Tuple, Any

class ProductDimSolutions:
    """Mixin class for Product over dimension problem solutions."""

    def reference_solution(self, input_tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """
        PyTorch implementation of product over dimension.

        Args:
            input_tensor: Input tensor of arbitrary shape
            dim: Dimension to reduce over

        Returns:
            Result of product reduction with keepdim=True
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            return torch.prod(input_tensor, dim=dim, keepdim=True)
