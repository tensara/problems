import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any

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
