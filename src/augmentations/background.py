"""Procedural smear-background synthesis and mask-guided background randomisation (component C3)."""
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from src.augmentations.transforms import IMAGENET_MEAN, IMAGENET_STD

# base tints of a Romanowsky-stained smear
PLASMA_BASE = (0.90, 0.85, 0.93)  # pale lavender plasma
RBC_BASE = (0.84, 0.60, 0.66)  # eosin pink erythrocyte
BRIGHTNESS = (0.55, 1.10)  # overall brightness
CHANNEL_JITTER = 0.05  # small hue wobble around the base
_LOWRES = 56  # erythrocytes are drawn here, then upsampled
RBC_CHUNK = 32
# Field scale (MorphoMix C2' small cell and C4 field, `field_rbc` per image): Aria erythrocytes are ~15-20 px across at
# 224 (white cells ~27 px), so radius 0.065-0.09 of the half side, drawn on a finer grid to keep the central pallor
FIELD_RBC_RADIUS = (0.065, 0.09)
FIELD_GRID = 112


def _uniform(shape, low: float, high: float, device, dtype) -> torch.Tensor:
    return torch.rand(shape, device=device, dtype=dtype) * (high - low) + low


def _tint(batch: int, base: Tuple[float, float, float], device, dtype,
          brightness: Tuple[float, float] = BRIGHTNESS) -> torch.Tensor:
    """A plausible plasma tint."""
    base_t = torch.tensor(base, device=device, dtype=dtype).view(1, 3, 1, 1)
    factor = _uniform((batch, 1, 1, 1), brightness[0], brightness[1], device, dtype)
    jitter = _uniform((batch, 3, 1, 1), -CHANNEL_JITTER, CHANNEL_JITTER, device, dtype)
    return torch.clamp(base_t * factor + jitter, 0.0, 1.0)


# below this luminance a pixel is surround, not cell
CELL_LUMINANCE_FLOOR = 0.10


def synthesize_backgrounds(
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    n_rbc: Tuple[int, int] = (5, 25),
    illumination: float = 0.18,
    rbc_radius: Tuple[float, float] = (0.07, 0.16),
    grid: int = _LOWRES,
) -> torch.Tensor:
    """Returns [B, 3, H, W] reflectance in [0, 1]; `rbc_radius` in units of half the side, erythrocytes drawn on a
    `grid` x `grid` canvas."""
    plasma = _tint(batch, PLASMA_BASE, device, dtype)
    bg = plasma.expand(batch, 3, height, width).clone()

    # Low-frequency illumination gradient
    field = _uniform((batch, 1, 4, 4), 1.0 - illumination, 1.0 + illumination, device, dtype)
    bg = bg * F.interpolate(field, size=(height, width), mode="bilinear", align_corners=False)

    # Elliptical erythrocytes drawn at low resolution, then upsampled
    k = int(torch.randint(n_rbc[0], n_rbc[1] + 1, (1,), device=device).item())
    if k > 0:
        ys = torch.linspace(-1.0, 1.0, grid, device=device, dtype=dtype).view(1, 1, grid, 1)
        xs = torch.linspace(-1.0, 1.0, grid, device=device, dtype=dtype).view(1, 1, 1, grid)
        cy = _uniform((batch, k, 1, 1), -1.0, 1.0, device, dtype)
        cx = _uniform((batch, k, 1, 1), -1.0, 1.0, device, dtype)
        ry = _uniform((batch, k, 1, 1), rbc_radius[0], rbc_radius[1], device, dtype)
        rx = ry * _uniform((batch, k, 1, 1), 0.75, 1.3, device, dtype)
        theta = _uniform((batch, k, 1, 1), 0.0, 3.14159, device, dtype)

        # running maximum over chunks of RBC_CHUNK erythrocytes: exact, and a dense field stays small in memory
        alpha = torch.zeros((batch, 1, grid, grid), device=device, dtype=dtype)
        for j in range(0, k, RBC_CHUNK):
            c = slice(j, j + RBC_CHUNK)
            dy, dx = ys - cy[:, c], xs - cx[:, c]
            cos_t, sin_t = torch.cos(theta[:, c]), torch.sin(theta[:, c])
            u = (dx * cos_t + dy * sin_t) / rx[:, c]
            v = (-dx * sin_t + dy * cos_t) / ry[:, c]
            # Soft-edged discs with a paler centre
            disc = torch.sigmoid((1.0 - (u * u + v * v)) * 8.0)
            pallor = 1.0 - 0.35 * torch.sigmoid((0.35 - (u * u + v * v)) * 10.0)
            alpha = torch.maximum(alpha, (disc * pallor).amax(dim=1, keepdim=True))
        alpha = alpha.clamp(0.0, 1.0) * _uniform((batch, 1, 1, 1), 0.45, 0.85, device, dtype)
        alpha = F.interpolate(alpha, size=(height, width), mode="bilinear", align_corners=False)

        rbc = _tint(batch, RBC_BASE, device, dtype)
        bg = bg * (1.0 - alpha) + rbc * alpha

    grain = torch.randn((batch, 1, height, width), device=device, dtype=dtype) * 0.012
    return torch.clamp(bg + grain, 0.0, 1.0)


def apply_background_randomization(
    images: torch.Tensor,
    masks: torch.Tensor,
    prob: float = 0.5,
    feather_kernel: int = 5,
    n_rbc: Tuple[int, int] = (5, 25),
    dark_threshold: float = CELL_LUMINANCE_FLOOR,
    fields: Optional[torch.Tensor] = None,
    field_rbc: Tuple[int, int] = (40, 120),
) -> torch.Tensor:
    """Replace the non-cell region with a synthetic background (C3). `fields` [B] bool marks the samples rendered at
    field scale (MorphoMix C2' small cell, C4): they always get a background, a dense one of `field_rbc` small
    erythrocytes; it is drawn for the whole batch whenever `fields` is given, so the RNG stream does not depend on
    the data."""
    b, _, h, w = images.shape
    device, dtype = images.device, images.dtype
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1)

    rgb = images * std + mean

    # keep = cell pixels above the luminance floor
    luminance = rgb.mean(dim=1, keepdim=True)
    keep = masks * (luminance >= dark_threshold).to(masks.dtype)
    # Give the cell back its own dark boundary
    keep = torch.minimum(F.max_pool2d(keep, kernel_size=3, stride=1, padding=1), masks)

    if feather_kernel > 1:
        pad = feather_kernel // 2
        soft = F.avg_pool2d(keep, kernel_size=feather_kernel, stride=1, padding=pad)
        # Clamp the soft edge to the hard region
        soft = torch.minimum(soft, keep)
    else:
        soft = keep

    has_cell = (keep.flatten(1).sum(dim=1) > 0).view(b, 1, 1, 1)
    selected = (torch.rand((b, 1, 1, 1), device=device) < prob) & has_cell
    # drawn before the early return, so the RNG stream does not depend on the masks
    bg = synthesize_backgrounds(b, h, w, device=device, dtype=dtype, n_rbc=n_rbc)
    if fields is not None:
        field = fields.view(b, 1, 1, 1)
        selected = selected | (field & has_cell)
        bg = torch.where(field, synthesize_backgrounds(b, h, w, device=device, dtype=dtype, n_rbc=field_rbc,
                                                       rbc_radius=FIELD_RBC_RADIUS, grid=FIELD_GRID), bg)
    if not bool(selected.any()):
        return images
    composed = (torch.clamp(rgb * soft + bg * (1.0 - soft), 0.0, 1.0) - mean) / std
    return torch.where(selected, composed, images)
