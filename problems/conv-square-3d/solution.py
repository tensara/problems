import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any
from tinygrad.tensor import Tensor

class ConvSquare3dSolutions:
    """Mixin class for 3D convolution problem with square input and square kernel solutions."""

    def reference_solution(self, input_volume: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of 3D convolution with square input and square kernel.

        Args:
            input_volume: Input volume tensor of shape (D, H, W)
            kernel: Convolution kernel tensor of shape (K, K, K) where K is the kernel size

        Returns:
            Result of convolution with zero padding
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            assert kernel.size(0) == kernel.size(1) == kernel.size(2), "Kernel must be cubic (equal dimensions)"

            input_reshaped = input_volume.view(1, 1, input_volume.size(0), input_volume.size(1), input_volume.size(2))
            kernel_reshaped = kernel.view(1, 1, kernel.size(0), kernel.size(1), kernel.size(2))

            padding = kernel.size(0) // 2

            result = F.conv3d(
                input_reshaped,
                kernel_reshaped,
                padding=padding
            )

            return result.view(input_volume.size(0), input_volume.size(1), input_volume.size(2))

    def reference_tinygrad_solution(self, input_volume: Tensor, kernel: Tensor) -> Tensor:
        """
        Tinygrad implementation of 3D convolution with square input and square kernel.

        Args:
            input_volume: Input volume tensor of shape (D, H, W)
            kernel: Convolution kernel tensor of shape (K, K, K) where K is the kernel size

        Returns:
            Result of convolution with zero padding
        """
        assert kernel.shape[0] == kernel.shape[1] == kernel.shape[2], "Kernel must be cubic (equal dimensions)"

        # Convert to shape expected by conv3d: [batch, channels, depth, height, width]
        input_reshaped = input_volume.reshape(1, 1, input_volume.shape[0], input_volume.shape[1], input_volume.shape[2])
        kernel_reshaped = kernel.reshape(1, 1, kernel.shape[0], kernel.shape[1], kernel.shape[2])

        padding = kernel.shape[0] // 2

        result = input_reshaped.conv3d(
            weight=kernel_reshaped,
            padding=padding
        )

        # Reshape back to original dimensions
        return result.reshape(input_volume.shape[0], input_volume.shape[1], input_volume.shape[2])
