import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any
from tinygrad.tensor import Tensor

class MaxPool2dSolutions:
    """Mixin class for 2D max pooling problem solutions."""

    def reference_solution(self, input_tensor: torch.Tensor, kernel_size: int,
                         stride: int, padding: int, dilation: int) -> torch.Tensor:
        """
        PyTorch implementation of 2D max pooling.

        Args:
            input_tensor: Input tensor of shape (H, W)
            kernel_size: Size of the pooling window
            stride: Stride of the pooling window
            padding: Padding to be applied before pooling
            dilation: Spacing between kernel elements

        Returns:
            Result of max pooling
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            input_reshaped = input_tensor.view(1, 1, input_tensor.size(0), input_tensor.size(1))

            result = F.max_pool2d(
                input_reshaped,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            )

            return result.view(result.size(2), result.size(3))

    def reference_tinygrad_solution(self, input_tensor: Tensor, kernel_size: int,
                                  stride: int, padding: int, dilation: int) -> Tensor:
        """
        Tinygrad implementation of 2D max pooling.

        Args:
            input_tensor: Input tensor of shape (H, W)
            kernel_size: Size of the pooling window
            stride: Stride of the pooling window
            padding: Padding to be applied before pooling
            dilation: Spacing between kernel elements

        Returns:
            Result of max pooling
        """
        # The input shape for Tinygrad's max_pool2d is expected to be (N, C, H, W) or (C, H, W)
        # The problem defines input as (H, W), so we need to reshape it to (1, 1, H, W)
        input_reshaped = input_tensor.reshape(1, 1, input_tensor.shape[0], input_tensor.shape[1])

        result = input_reshaped.max_pool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )

        # Reshape back to (H_out, W_out)
        return result.reshape(result.shape[2], result.shape[3])
