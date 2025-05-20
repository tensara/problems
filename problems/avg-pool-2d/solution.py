import torch
import torch.nn.functional as F
from tinygrad.tensor import Tensor
from typing import List, Dict, Tuple, Any

class AvgPool2dSolutions:
    """Mixin class for 2D average pooling problem solutions."""

    def reference_solution(self, input_tensor: torch.Tensor, kernel_size: int,
                         stride: int, padding: int) -> torch.Tensor:
        """
        PyTorch implementation of 2D average pooling.

        Args:
            input_tensor: Input tensor of shape (H, W)
            kernel_size: Size of the pooling window
            stride: Stride of the pooling window
            padding: Padding to be applied before pooling

        Returns:
            Result of average pooling
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            input_reshaped = input_tensor.view(1, 1, input_tensor.size(0), input_tensor.size(1))

            result = F.avg_pool2d(
                input_reshaped,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )

            return result.view(result.size(2), result.size(3))

    def reference_tinygrad_solution(self, input_tensor: Tensor, kernel_size: int,
                                  stride: int, padding: int) -> Tensor:
        """
        Tinygrad implementation of 2D average pooling.

        Args:
            input_tensor: Input tensor of shape (H, W)
            kernel_size: Size of the pooling window
            stride: Stride of the pooling window
            padding: Padding to be applied before pooling

        Returns:
            Result of average pooling
        """
        # Tinygrad's avg_pool2d is the direct equivalent of PyTorch's avg_pool2d
        # The input shape for Tinygrad's avg_pool2d is expected to be (N, C, H, W) or (C, H, W)
        # The problem defines input as (H, W), so we need to reshape it to (1, 1, H, W)
        input_reshaped = input_tensor.reshape(1, 1, input_tensor.shape[0], input_tensor.shape[1])

        result = input_reshaped.avg_pool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # Reshape back to (H_out, W_out)
        return result.reshape(result.shape[2], result.shape[3])
