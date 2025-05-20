import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any
from tinygrad.tensor import Tensor

class Conv2dSolutions:
    """Mixin class for 2D convolution problem solutions."""

    def reference_solution(self, input_image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of 2D convolution.

        Args:
            input_image: Input image tensor of shape (H, W)
            kernel: Convolution kernel tensor of shape (Kh, Kw)

        Returns:
            Result of convolution with zero padding
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            # Ensure kernel sizes are odd
            assert kernel.size(0) % 2 == 1, "Kernel height must be odd"
            assert kernel.size(1) % 2 == 1, "Kernel width must be odd"

            # Perform 2D convolution using PyTorch's built-in function
            # Convert to shape expected by conv2d: [batch, channels, height, width]
            input_reshaped = input_image.view(1, 1, input_image.size(0), input_image.size(1))
            kernel_reshaped = kernel.view(1, 1, kernel.size(0), kernel.size(1))

            # Calculate padding size to maintain the same output size
            padding_h = kernel.size(0) // 2
            padding_w = kernel.size(1) // 2

            # Perform convolution
            result = F.conv2d(
                input_reshaped,
                kernel_reshaped,
                padding=(padding_h, padding_w)
            )

            # Reshape back to original dimensions
            return result.view(input_image.size(0), input_image.size(1))

    def reference_tinygrad_solution(self, input_image: Tensor, kernel: Tensor) -> Tensor:
        """
        Tinygrad implementation of 2D convolution.

        Args:
            input_image: Input image tensor of shape (H, W)
            kernel: Convolution kernel tensor of shape (Kh, Kw)

        Returns:
            Result of convolution with zero padding
        """
        # Ensure kernel sizes are odd
        assert kernel.shape[0] % 2 == 1, "Kernel height must be odd"
        assert kernel.shape[1] % 2 == 1, "Kernel width must be odd"

        # Convert to shape expected by conv2d: [batch, channels, height, width]
        input_reshaped = input_image.reshape(1, 1, input_image.shape[0], input_image.shape[1])
        kernel_reshaped = kernel.reshape(1, 1, kernel.shape[0], kernel.shape[1])

        # Calculate padding size to maintain the same output size
        padding_h = kernel.shape[0] // 2
        padding_w = kernel.shape[1] // 2

        # Perform convolution
        result = input_reshaped.conv2d(
            weight=kernel_reshaped,
            padding=(padding_h, padding_w)
        )

        # Reshape back to original dimensions
        return result.reshape(input_image.shape[0], input_image.shape[1])
