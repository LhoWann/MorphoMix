"""Deterministic 8-view test-time augmentation."""
from typing import List

import torch


def dihedral_views(images: torch.Tensor) -> List[torch.Tensor]:
    """The 8 symmetry views of a [B, 3, H, W] batch (square images only)."""
    views = []
    for flip in (False, True):
        base = torch.flip(images, dims=[3]) if flip else images
        for k in range(4):
            views.append(torch.rot90(base, k=k, dims=[2, 3]) if k else base)
    return views


@torch.no_grad()
def tta_logits(model, images: torch.Tensor, n_views: int = 8) -> torch.Tensor:
    """Mean softmax probability over the first `n_views` dihedral views."""
    if n_views is None or n_views <= 1:
        return torch.log_softmax(model(images).float(), dim=-1)

    # flips and rot90 return strided views; copy each in the caller's memory format (channels_last on the GPU path)
    fmt = torch.channels_last if images.is_contiguous(memory_format=torch.channels_last) else torch.contiguous_format
    views = dihedral_views(images)[:n_views]
    probs = None
    for view in views:
        p = torch.softmax(model(view.contiguous(memory_format=fmt)).float(), dim=-1)
        probs = p if probs is None else probs + p
    probs = probs / len(views)
    return torch.log(probs.clamp_min(1e-12))
