import torch
from typing import List, Dict, Tuple, Any

class GemmReluSolutions:
    """Mixin class for GEMM with Bias and ReLU problem solutions."""

    def reference_solution(self, input_matrix: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of GEMM with bias and ReLU.

        Args:
            input_matrix: Input matrix of shape (B, N) (batch_size, input_features)
            weights: Weight matrix of shape (M, N) (output_features, input_features)
            bias: Bias vector of shape (M) (output_features)

        Returns:
            Result of ReLU(input_matrix @ weights.T + bias)
        """

        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            # Matrix multiplication: (B, N) @ (N, M) -> (B, M)
            result = torch.mm(input_matrix, weights.t())
            # Add bias: (B, M) + (M) -> (B, M)
            result = result + bias
            # Apply ReLU activation
            result = torch.relu(result)

            return result
