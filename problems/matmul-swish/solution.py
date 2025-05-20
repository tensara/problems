import torch
from typing import List, Dict, Tuple, Any

class MatmulSwishSolutions:
    """Mixin class for matrix multiplication with Swish activation problem solutions."""

    def reference_solution(self, input_matrix: torch.Tensor, weight_matrix: torch.Tensor,
                         bias: torch.Tensor, scaling_factor: float) -> torch.Tensor:
        """
        PyTorch implementation of matrix multiplication with Swish activation.

        Args:
            input_matrix: Input tensor of shape (batch_size, in_features)
            weight_matrix: Weight tensor of shape (out_features, in_features)
            bias: Bias tensor of shape (out_features,)
            scaling_factor: Scaling factor to apply after Swish activation

        Returns:
            Result of matrix multiplication with Swish activation and scaling
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            # Linear transformation
            z = torch.matmul(input_matrix, weight_matrix.t()) + bias

            # Swish activation: x * sigmoid(x)
            output = z * torch.sigmoid(z)

            # Apply scaling
            output = output * scaling_factor

            return output
