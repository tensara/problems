import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any
from tinygrad.tensor import Tensor

class CosineSimilaritySolutions:
    """Mixin class for Cosine Similarity problem solutions."""

    def reference_solution(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of element-wise Cosine Similarity.

        Args:
            predictions: Predictions tensor of shape (N, D)
            targets: Targets tensor of shape (N, D)

        Returns:
            Negative cosine similarity tensor of shape (N,)
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            return 1 - F.cosine_similarity(predictions, targets, dim=1)

    def reference_tinygrad_solution(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Tinygrad implementation of element-wise Cosine Similarity.

        Args:
            predictions: Predictions tensor of shape (N, D)
            targets: Targets tensor of shape (N, D)

        Returns:
            Negative cosine similarity tensor of shape (N,)
        """
        # Calculate dot product along dimension 1
        dot_product = (predictions * targets).sum(axis=1)

        # Calculate magnitudes along dimension 1
        predictions_magnitude = predictions.pow(2).sum(axis=1).sqrt()
        targets_magnitude = targets.pow(2).sum(axis=1).sqrt()

        # Calculate cosine similarity
        cosine_sim = dot_product / (predictions_magnitude * targets_magnitude)

        # Return negative cosine similarity
        return 1 - cosine_sim
