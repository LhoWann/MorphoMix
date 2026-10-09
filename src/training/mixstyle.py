"""MixStyle (Zhou et al., ICLR 2021) on the ConvNeXt stages, training only."""
from typing import List, Sequence

import torch
import torch.nn as nn


class MixStyle:
    """Forward hook that mixes feature statistics between the samples of a training batch.

    With probability `p` per hooked stage and batch, each sample's channel-wise mean and sd over H x W are replaced by
    lambda * own + (1 - lambda) * those of a random other sample of the batch, lambda ~ Beta(alpha, alpha) per sample;
    the statistics are detached (the paper's random-shuffle variant: one training domain). Every draw uses the torch
    RNG of the feature device; in eval mode the hook is a no-op, so validation and test scoring never see it.
    """

    def __init__(self, p: float, alpha: float = 0.1, eps: float = 1e-6):
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = eps
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def attach(self, stages: Sequence[nn.Module]) -> "MixStyle":
        self.handles = [stage.register_forward_hook(self) for stage in stages]
        return self

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __call__(self, module: nn.Module, inputs, output: torch.Tensor):
        if not module.training or self.p <= 0:
            return None
        device = output.device
        if torch.rand((), device=device) >= self.p:
            return None
        b = output.shape[0]
        x = output.float()
        mu = x.mean(dim=(2, 3), keepdim=True).detach()
        sig = (x.var(dim=(2, 3), keepdim=True) + self.eps).sqrt().detach()
        concentration = torch.full((b, 1, 1, 1), self.alpha, device=device)
        lam = torch.distributions.Beta(concentration, concentration).sample()
        perm = torch.randperm(b, device=device)
        mu_mix = lam * mu + (1 - lam) * mu[perm]
        sig_mix = lam * sig + (1 - lam) * sig[perm]
        return ((x - mu) / sig * sig_mix + mu_mix).to(output.dtype)
