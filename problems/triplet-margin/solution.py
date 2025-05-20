import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Any

class TripletMarginSolutions:
    """Mixin class for Triplet Margin Loss problem solutions."""

    margin = 1.0  # Standard margin for TripletMarginLoss

    def reference_solution(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of Triplet Margin Loss.

        Args:
            anchor (torch.Tensor): Anchor points, shape (batch_size, embedding_dim)
            positive (torch.Tensor): Positive examples, shape (batch_size, embedding_dim)
            negative (torch.Tensor): Negative examples, shape (batch_size, embedding_dim)

        Returns:
            torch.Tensor: Triplet loss value (scalar)
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
            # Use PyTorch's built-in TripletMarginLoss
            loss_fn = nn.TripletMarginLoss(margin=self.margin)

            # Move to the same device as inputs
            loss_fn = loss_fn.to(anchor.device)

            # Calculate the loss
            loss = loss_fn(anchor, positive, negative)

            return loss
