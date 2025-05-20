import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any

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
