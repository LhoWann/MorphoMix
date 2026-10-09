"""The Azure-B cell prior as a [B, 1, H, W] map: the source of the one cell mask every MorphoMix component uses."""
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import torch

from src.cam.blast_prior import prior_tensor_from_rgb
from src.augmentations.transforms import denormalize_batch


class PriorOnlyCAM:
    """Returns the Azure-B soft cell envelope, normalised per sample, ignoring the model."""

    def __init__(self, prior_threads: int = 8):
        self._pool = ThreadPoolExecutor(max_workers=max(1, prior_threads)) if prior_threads > 1 \
            else None

    def compute_priors(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """[B, 1, H, W] float32 priors on the input device, from normalized images."""
        rgb_batch = denormalize_batch(input_tensor)
        images = [rgb_batch[b] for b in range(rgb_batch.shape[0])]
        fn = prior_tensor_from_rgb
        priors = list(self._pool.map(fn, images)) if self._pool is not None \
            else [fn(im) for im in images]
        return torch.from_numpy(np.stack(priors)).to(input_tensor.device, dtype=torch.float32)

    @torch.no_grad()
    def generate(self, input_tensor: torch.Tensor, priors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """[B, 1, H, W] soft cell envelope in [0, 1]."""
        if priors is None:
            priors = self.compute_priors(input_tensor)
        cell = priors[:, 0:1]
        c_min = cell.amin(dim=(2, 3), keepdim=True)
        c_max = cell.amax(dim=(2, 3), keepdim=True)
        return ((cell - c_min) / (c_max - c_min).clamp(min=1e-8)).clamp(0.0, 1.0)

    def remove_hooks(self) -> None:
        """No hooks are ever registered."""
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
