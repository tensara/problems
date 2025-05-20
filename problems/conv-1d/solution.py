import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any
from tinygrad.tensor import Tensor

class Conv1dSolutions:
    """Mixin class for 1D convolution problem solutions."""

    def reference_solution(self, input_signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of 1D convolution.

        Args:
            input_signal: Input signal tensor of shape (N)
            kernel: Convolution kernel tensor of shape (K)

        Returns:
            Result of convolution with zero padding
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            # Ensure kernel size is odd
            assert kernel.size(0) % 2 == 1, "Kernel size must be odd"

            # Perform 1D convolution using PyTorch's built-in function
            # Convert to shape expected by conv1d: [batch, channels, length]
            input_reshaped = input_signal.view(1, 1, -1)
            kernel_reshaped = kernel.view(1, 1, -1)

            # Calculate padding size to maintain the same output size
            padding = kernel.size(0) // 2

            # Perform convolution
            result = F.conv1d(
                input_reshaped,
                kernel_reshaped,
                padding=padding
            )

            # Reshape back to original dimensions
            return result.view(-1)

    def reference_tinygrad_solution(self, input_signal: Tensor, kernel: Tensor) -> Tensor:
        """
        Tinygrad implementation of 1D convolution.

        Args:
            input_signal: Input signal tensor of shape (N)
            kernel: Convolution kernel tensor of shape (K)

        Returns:
            Result of convolution with zero padding
        """
        # Ensure kernel size is odd
        assert kernel.shape[0] % 2 == 1, "Kernel size must be odd"

        # Convert to shape expected by conv1d: [batch, channels, length]
        input_reshaped = input_signal.reshape(1, 1, -1)
        kernel_reshaped = kernel.reshape(1, 1, -1)

        # Calculate padding size to maintain the same output size
        padding = kernel.shape[0] // 2

        # Perform convolution
        result = input_reshaped.conv1d(
            weight=kernel_reshaped,
            padding=padding
        )

        # Reshape back to original dimensions
        return result.reshape(-1)
