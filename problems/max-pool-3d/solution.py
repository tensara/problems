import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any
from tinygrad.tensor import Tensor

class MaxPool3dSolutions:
    """Mixin class for 3D max pooling problem solutions."""

    def reference_solution(self, input_tensor: torch.Tensor, kernel_size: int,
                         stride: int, padding: int, dilation: int) -> torch.Tensor:
        """
        PyTorch implementation of 3D max pooling.

        Args:
            input_tensor: Input tensor of shape (H, W, D)
            kernel_size: Size of the pooling window
            stride: Stride of the pooling window
            padding: Padding to be applied before pooling
            dilation: Spacing between kernel elements

        Returns:
            Result of max pooling
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            input_reshaped = input_tensor.view(1, 1, input_tensor.size(0), input_tensor.size(1), input_tensor.size(2))

            result = F.max_pool3d(
                input_reshaped,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            )

            return result.view(result.size(2), result.size(3), result.size(4))

    def reference_tinygrad_solution(self, input_tensor: Tensor, kernel_size: int,
                                  stride: int, padding: int, dilation: int) -> Tensor:
        """
        Tinygrad implementation of 3D max pooling.

        Args:
            input_tensor: Input tensor of shape (H, W, D)
            kernel_size: Size of the pooling window
            stride: Stride of the pooling window
            padding: Padding to be applied before pooling
            dilation: Spacing between kernel elements

        Returns:
            Result of max pooling
        """
        # The input shape for Tinygrad's max_pool3d is expected to be (N, C, D, H, W) or (C, D, H, W)
        # The problem defines input as (H, W, D), so we need to reshape it to (1, 1, H, W, D)
        # and then permute to (1, 1, D, H, W) to match Tinygrad's expected input shape.
        input_reshaped = input_tensor.reshape(1, 1, input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2])
        input_permuted = input_reshaped.permute(0, 1, 4, 2, 3) # Permute to (N, C, D, H, W)

        result = input_permuted.max_pool3d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )

        # Permute back and reshape to (H_out, W_out, D_out)
        result_permuted_back = result.permute(0, 1, 3, 4, 2) # Permute back to (N, C, H_out, W_out, D_out)
        return result_permuted_back.reshape(result_permuted_back.shape[2], result_permuted_back.shape[3], result_permuted_back.shape[4])
