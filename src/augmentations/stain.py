"""Colour code of MorphoMix C1 and the MLL23 style bank: CIE Lab, the nucleus / cytoplasm split, per-region Lab
moments and per-region Reinhard matching, on batches of RGB in [0, 1], [B, 3, H, W] (`to_unit` / `from_unit` convert
ImageNet-normalised batches)."""
from typing import Sequence, Tuple

import torch
import torch.nn.functional as nnf

from src.augmentations.transforms import IMAGENET_MEAN, IMAGENET_STD

REGIONS = ("nucleus", "cytoplasm")
MIN_REGION_PX = 64  # a region below this takes the whole cell's moments; a bank row needs both regions above it
SD_FLOOR = 1.0  # Lab units: a nearly flat region is not stretched without bound
# Reinhard sd ratio cap per channel: ref / own sd has p90 ~3.3-3.7 on C-NMC cells (partly resampling, MLL23 288 px vs
# C-NMC 450 px -> 224), and an uncapped ratio amplifies nucleus texture and chroma noise about 4x
MAX_SD_SCALE = 3.0
SEAM_PX = 3  # box size of the nucleus / cytoplasm seam: a 1 px blend on either side of the region border

# sRGB (D65) -> XYZ, and the D65 white point
_RGB_TO_XYZ = torch.tensor([[0.412453, 0.357580, 0.180423],
                            [0.212671, 0.715160, 0.072169],
                            [0.019334, 0.119193, 0.950227]])
_WHITE = torch.tensor([0.950456, 1.0, 1.088754])
_XYZ_TO_RGB = torch.linalg.inv(_RGB_TO_XYZ)


def to_unit(images: torch.Tensor) -> torch.Tensor:
    """ImageNet-normalised batch -> RGB in [0, 1] (float32)."""
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    return (images.float() * std + mean).clamp(0.0, 1.0)


def from_unit(rgb: torch.Tensor) -> torch.Tensor:
    """Inverse of `to_unit`."""
    mean = torch.tensor(IMAGENET_MEAN, device=rgb.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=rgb.device).view(1, 3, 1, 1)
    return (rgb.clamp(0.0, 1.0) - mean) / std


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """[B, 3, H, W] sRGB in [0, 1] -> CIE Lab (L in [0, 100]), as OpenCV's float conversion."""
    rgb = rgb.clamp(0.0, 1.0)
    lin = torch.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    xyz = torch.einsum("ij,bjhw->bihw", _RGB_TO_XYZ.to(rgb), lin) / _WHITE.to(rgb).view(1, 3, 1, 1)
    eps = (6.0 / 29.0) ** 3
    f = torch.where(xyz > eps, xyz.clamp(min=eps) ** (1.0 / 3.0), xyz / (3 * (6.0 / 29.0) ** 2) + 4.0 / 29.0)
    fx, fy, fz = f.unbind(dim=1)
    return torch.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], dim=1)


def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    """Inverse of `rgb_to_lab`: [B, 3, H, W] CIE Lab -> sRGB clamped to [0, 1] (out-of-gamut colours clip)."""
    fy = (lab[:, 0] + 16.0) / 116.0
    f = torch.stack([fy + lab[:, 1] / 500.0, fy, fy - lab[:, 2] / 200.0], dim=1)
    delta = 6.0 / 29.0
    xyz = torch.where(f > delta, f ** 3, 3 * delta ** 2 * (f - 4.0 / 29.0)) * _WHITE.to(lab).view(1, 3, 1, 1)
    lin = torch.einsum("ij,bjhw->bihw", _XYZ_TO_RGB.to(lab), xyz).clamp(0.0, 1.0)
    return torch.where(lin > 0.0031308, 1.055 * lin ** (1.0 / 2.4) - 0.055, 12.92 * lin).clamp(0.0, 1.0)


@torch.no_grad()
def split_nucleus(rgb: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """Nucleus [B, 1, H, W] bool: cell pixels above the Otsu threshold of HSV saturation inside the cell.

    The nucleus is far more saturated than the cytoplasm in both labs (MLL23: ~170-190 vs ~65-110 of 255), so the
    rest of the cell is the cytoplasm. Otsu runs on the sorted in-cell values (exact, no histogram, deterministic on
    CUDA) and only splits between distinct values, so `sat > threshold` realises the chosen split.
    """
    b, _, h, w = rgb.shape
    hi, lo = rgb.amax(dim=1), rgb.amin(dim=1)
    sat = ((hi - lo) / hi.clamp(min=1e-6)).flatten(1).double()
    inside = (cell[:, 0] > 0.5).flatten(1)
    n = inside.sum(dim=1, keepdim=True).double()
    v = torch.where(inside, sat, torch.full_like(sat, float("inf"))).sort(dim=1).values
    csum = torch.where(torch.isfinite(v), v, torch.zeros_like(v)).cumsum(dim=1)
    k = torch.arange(1, h * w + 1, device=rgb.device, dtype=torch.float64).unsqueeze(0)  # pixels in the low class
    w0 = k / n.clamp(min=1)
    between = w0 * (1 - w0) * (csum / k - (csum[:, -1:] - csum) / (n - k).clamp(min=1)) ** 2
    boundary = torch.cat([v[:, 1:] > v[:, :-1], torch.zeros_like(v[:, :1], dtype=torch.bool)], dim=1)
    between = torch.where(boundary & (k < n), between, torch.full_like(between, -1.0))
    threshold = v.gather(1, between.argmax(dim=1, keepdim=True))
    return (inside & (sat > threshold)).view(b, 1, h, w)


def region_moments(lab: torch.Tensor, regions: Sequence[torch.Tensor]
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean and sd [B, R, 3] of Lab [B, 3, H, W] over each bool region [B, 1, H, W], and pixel counts [B, R]."""
    flat = lab.flatten(2)
    means, sds, areas = [], [], []
    for region in regions:
        r = region.flatten(2).to(flat.dtype)
        area = r.sum(dim=-1)
        mean = (flat * r).sum(dim=-1) / area.clamp(min=1)
        var = ((flat - mean.unsqueeze(-1)) ** 2 * r).sum(dim=-1) / area.clamp(min=1)
        means.append(mean)
        sds.append(var.sqrt())
        areas.append(area[:, 0])
    return torch.stack(means, dim=1), torch.stack(sds, dim=1), torch.stack(areas, dim=1)


def cell_moments(rgb: torch.Tensor, cell: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Nucleus and cytoplasm Lab mean and sd [B, 2, 3] of the cells in `cell` [B, 1, H, W], and their nuclei.

    A region below MIN_REGION_PX (a C-NMC blast with almost no cytoplasm) takes the moments of the whole cell.
    """
    inside = cell > 0.5
    nucleus = split_nucleus(rgb, inside.float())
    mean, sd, area = region_moments(rgb_to_lab(rgb.float()), (nucleus, inside & ~nucleus, inside))
    small = (area[:, :2] < MIN_REGION_PX).unsqueeze(-1)
    mean = torch.where(small, mean[:, 2:], mean[:, :2])
    sd = torch.where(small, sd[:, 2:], sd[:, :2])
    return mean, sd, nucleus


@torch.no_grad()
def match_regions(rgb: torch.Tensor, cell: torch.Tensor, ref_mean: torch.Tensor, ref_sd: torch.Tensor
                  ) -> torch.Tensor:
    """MorphoMix C1: per region and Lab channel x -> (x - mean) / sd * sd_ref + mean_ref (Reinhard et al., 2001, on
    the nucleus and the cytoplasm separately), towards reference moments [B, 2, 3], with sd_ref / sd capped at
    MAX_SD_SCALE.

    The two maps are mixed by the nucleus indicator box-filtered over SEAM_PX px, so the weight is 0 / 1 except on a
    narrow seam: a hard split prints speckle outlines as sharp edges, while a wide blend spreads the thin C-NMC
    cytoplasm ring onto the nucleus map. Pixels outside the cell are returned unchanged.
    """
    mean, sd, nucleus = cell_moments(rgb, cell)
    lab = rgb_to_lab(rgb.float())
    scale = (ref_sd / sd.clamp(min=SD_FLOOR)).clamp(max=MAX_SD_SCALE)
    mapped = [(lab - mean[:, i, :, None, None]) * scale[:, i, :, None, None] + ref_mean[:, i, :, None, None]
              for i in range(len(REGIONS))]
    w = nnf.avg_pool2d(nucleus.float(), SEAM_PX, stride=1, padding=SEAM_PX // 2, count_include_pad=False)
    return torch.where(cell > 0.5, lab_to_rgb(w * mapped[0] + (1 - w) * mapped[1]), rgb.float())
