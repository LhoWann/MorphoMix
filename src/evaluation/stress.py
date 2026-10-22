"""Stress suite: SELECTION families on the val cohort (LeukemiaAttri crops, real smear background)."""
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from src.augmentations.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.evaluation.tta import tta_logits


def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """[B, 3, H, W] RGB in [0, 1] -> HSV in [0, 1]."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mx, _ = rgb.max(dim=1)
    mn, _ = rgb.min(dim=1)
    d = mx - mn
    safe = torch.where(d > 0, d, torch.ones_like(d))
    h = torch.where(mx == r, ((g - b) / safe) % 6, torch.where(mx == g, (b - r) / safe + 2, (r - g) / safe + 4))
    h = torch.where(d > 0, h / 6.0, torch.zeros_like(h))
    s = torch.where(mx > 0, d / torch.where(mx > 0, mx, torch.ones_like(mx)), torch.zeros_like(mx))
    return torch.stack([h % 1.0, s, mx], dim=1)


def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    k = lambda n: (n + h * 6) % 6                      # noqa: E731
    f = lambda n: v - v * s * torch.clamp(torch.minimum(k(n), 4 - k(n)), 0, 1)   # noqa: E731
    return torch.stack([f(5), f(3), f(1)], dim=1)


def _gamma(g: float) -> Callable:
    return lambda x: x.clamp(1e-4, 1.0) ** g


def _bright(delta: float) -> Callable:
    return lambda x: (x + delta).clamp(0, 1)


def _hsv(dh: float = 0.0, ks: float = 1.0) -> Callable:
    def op(x):
        hsv = rgb_to_hsv(x)
        hsv = torch.stack([(hsv[:, 0] + dh) % 1.0, (hsv[:, 1] * ks).clamp(0, 1), hsv[:, 2]], dim=1)
        return hsv_to_rgb(hsv).clamp(0, 1)
    return op


def _zoom(s: float) -> Callable:
    """Zoom about the centre; a zoom-out reflects the smear into the border, since a constant pad would draw a
    frame (black would even mimic the C-NMC training background)."""
    def op(x):
        h = x.shape[-1]
        if s < 1:
            z = F.interpolate(x, scale_factor=s, mode="bilinear", align_corners=False)
            p = (h - z.shape[-1]) // 2
            return F.pad(z, (p, h - z.shape[-1] - p, p, h - z.shape[-2] - p), mode="reflect")
        c = int(round(h / s))
        p = (h - c) // 2
        return F.interpolate(x[..., p:p + c, p:p + c], size=h, mode="bilinear", align_corners=False)
    return op


# Global shifts only: on LeukemiaAttri the cell cannot be told from the smear by luminance (the C-NMC families
# `bright_bg*` swapped a black background and were the identity here), and the Azure-B mask also marks
# erythrocytes and is the mask MorphoMix trains on.
FAMILIES: Dict[str, Callable] = {
    "clean": lambda x: x,
    "gamma0.6": _gamma(0.6),
    "bright+0.2": _bright(0.2),
    "sat0.5": _hsv(ks=0.5),
    "hue+0.06": _hsv(dh=0.06),
    "zoom0.6": _zoom(0.6),
    "zoom1.4": _zoom(1.4),
}
SELECTION: List[str] = ["gamma0.6", "bright+0.2", "sat0.5", "hue+0.06", "zoom0.6", "zoom1.4"]


def load_cohort(dataset) -> tuple:
    """(uint8 [N, 3, H, W], labels [N]) from a BinaryLeukemiaDataset built with val transforms."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    xs, ys = [], []
    for i in range(len(dataset)):
        x, y = dataset[i][0], dataset[i][1]
        xs.append(((x * std + mean).clamp(0, 1) * 255).round().byte())
        ys.append(int(y))
    return torch.stack(xs), np.array(ys)


@torch.no_grad()
def evaluate_families(model: torch.nn.Module, images_u8: torch.Tensor, labels: np.ndarray,
                      families: Sequence[str], tta_views: int = 1, batch_size: int = 64
                      ) -> Dict[str, Dict[str, float]]:
    """Per family: ROC-AUC, macro-F1 at p(ALL) >= 0.5, predicted-positive rate (binary Normal=0 / ALL=1)."""
    device = next(model.parameters()).device
    model.eval()
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    out: Dict[str, Dict[str, float]] = {}
    for fam in families:
        op = FAMILIES[fam]
        probs = []
        for b in range(0, len(images_u8), batch_size):
            x = images_u8[b:b + batch_size].to(device).float() / 255.0
            x = op(x)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logp = tta_logits(model, (x - mean) / std, tta_views)
            probs.append(torch.exp(logp.float())[:, 1].cpu())
        p = torch.cat(probs).numpy()
        pred = (p >= 0.5).astype(int)
        out[fam] = {
            "roc_auc": float(roc_auc_score(labels, p)),
            "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
            "pos_rate": float(pred.mean()),
        }
    return out


def selection_score(per_family: Dict[str, Dict[str, float]], families: Sequence[str] = SELECTION) -> float:
    """Stress objective J for one model."""
    return float(np.mean([0.5 * per_family[f]["roc_auc"] + 0.5 * per_family[f]["macro_f1"] for f in families]))
