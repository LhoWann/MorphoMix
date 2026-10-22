"""Exact class-evidence map of the final ConvNeXt stage, as a CAM extractor (no gradients)."""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.factory import unwrap_model


def class_evidence_maps(head: nn.Module, feats: torch.Tensor) -> torch.Tensor:
    """Per-location share of every logit for timm's pool -> LayerNorm2d -> fc head: [B, K, h, w].

    LayerNorm of the pooled vector is linear in the channel-centred features once sigma (the channel std of
    the pooled vector) is fixed, so the mean of map k over locations equals logit k minus a constant.
    """
    f = feats.float()
    pooled = f.mean(dim=(2, 3))
    sigma = torch.sqrt(pooled.var(dim=1, unbiased=False) + head.norm.eps)
    weight = head.fc.weight.float() * head.norm.weight.float()
    return torch.einsum("bchw,kc->bkhw", f - f.mean(dim=1, keepdim=True), weight) / sigma.view(-1, 1, 1, 1)


class EvidenceCAM:
    """Positive part of the per-location decomposition of the target logit, upsampled and min-max normalised.

    The map averages exactly to the logit (up to a constant), so it shows where the classifier's evidence sits
    without the gradient approximations of Grad-CAM-style methods. Same interface as LayerCAM.
    """

    def __init__(self, model: nn.Module, chunk_size: int = 32):
        self.raw_model = unwrap_model(model)
        self.chunk_size = max(1, int(chunk_size))

    @torch.no_grad()
    def generate(self, input_tensor: torch.Tensor, target_class: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Heatmaps [B, 1, H, W] in [0, 1]."""
        out = []
        h, w = input_tensor.shape[-2:]
        for s in range(0, input_tensor.shape[0], self.chunk_size):
            x = input_tensor[s:s + self.chunk_size]
            feats = self.raw_model.forward_features(x)
            maps = class_evidence_maps(self.raw_model.head, feats)
            # the map means drop the head biases, so the predicted class comes from the logits
            y = (self.raw_model.forward_head(feats).argmax(dim=1) if target_class is None
                 else target_class[s:s + self.chunk_size])
            e = F.relu(maps.gather(1, y.view(-1, 1, 1, 1).expand(-1, 1, *maps.shape[-2:])))
            e = F.interpolate(e, size=(h, w), mode="bilinear", align_corners=False)
            lo = e.amin(dim=(2, 3), keepdim=True)
            span = e.amax(dim=(2, 3), keepdim=True) - lo
            out.append(torch.where(span > 1e-8, (e - lo) / (span + 1e-8), torch.zeros_like(e)))
        return torch.cat(out)

    def remove_hooks(self) -> None:
        """No hooks are registered."""
