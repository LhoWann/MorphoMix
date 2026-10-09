"""Class-weighted cross-entropy of the training loop."""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer("weight", weight)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(logits.device) if self.weight is not None else None
        # reduction="none" then mean, as in eval
        return F.cross_entropy(logits, targets, weight=w, reduction="none", label_smoothing=self.label_smoothing).mean()
