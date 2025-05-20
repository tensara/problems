import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any
from tinygrad.tensor import Tensor

class SoftmaxSolutions:
    """Mixin class for Softmax function problem solutions."""

    def reference_solution(self, input_tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """
        PyTorch implementation of softmax function.

        Args:
            input_tensor: Input tensor of arbitrary shape
            dim: Dimension to compute softmax over

        Returns:
            Softmax probabilities along the specified dimension
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            return F.softmax(input_tensor, dim=dim)

    def reference_tinygrad_solution(self, input_tensor: Tensor, dim: int) -> Tensor:
        """
        Tinygrad implementation of softmax function.

        Args:
            input_tensor: Input tensor of arbitrary shape
            dim: Dimension to compute softmax over

        Returns:
            Softmax probabilities along the specified dimension
        """
        # Softmax: exp(x) / sum(exp(x)) along the specified dimension
        exp_x = input_tensor.exp()
        return exp_x / exp_x.sum(axis=dim, keepdim=True)
