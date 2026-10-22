"""Stain-augmentation baselines on the GPU batch: HED jitter, RandStainNA and Stain Mix-up.

Each arm takes an ImageNet-normalised batch x [B, 3, H, W] and its foreground fg [B, 1, H, W] (bool, `foreground`)
and returns a batch of the same shape in which only foreground pixels of the selected samples change. C-NMC cells sit
on a black crop background that holds no stain: an optical-density or Lab transform there would colour it, a cue the
other arms never see. Every draw uses the global torch RNG of x's device, and the same number of draws is made
whatever the data, so a fixed seed stays bit-reproducible.
"""
import glob
import os
from typing import Dict, Tuple

import cv2
import numpy as np
import torch

from src.augmentations.background import CELL_LUMINANCE_FLOOR
from src.augmentations.stain import SD_FLOOR, from_unit, lab_to_rgb, region_moments, rgb_to_lab, to_unit

# Tellez et al. (IEEE TMI 2018; MedIA 2019, "HED-light"): alpha ~ U[1 - sigma, 1 + sigma], beta ~ U[-sigma, sigma]
HED_SIGMA = 0.05
# RandStainNA reference code (README usage): std_hyper -0.3, normal distribution, Lab
RANDSTAINNA_STD_HYPER = -0.3
RANDSTAINNA_KEYS = ("mean_avg", "mean_std", "sd_avg", "sd_std")
# Stain Mix-up reference code: concentrations scaled per stain by U[0.95, 1.05]
STAIN_INTENSITY = (0.95, 1.05)

# Ruifrok and Johnston (2001) H, E, DAB optical densities and log scale, exactly as skimage.color.rgb2hed / hed2rgb
_RGB_FROM_HED = torch.tensor([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]], dtype=torch.float64)
_HED_FROM_RGB = torch.linalg.inv(_RGB_FROM_HED)
_HED_LOG_ADJUST = float(np.log(1e-6))

# Macenko et al. (ISBI 2009): beta (OD floor in every channel) and alpha (robust angle extremes, in percent)
MACENKO_BETA = 0.15
MACENKO_ALPHA = 1.0
MIN_STAIN_PX = 64  # fewer usable pixels than this: no stain matrix, the sample is left unchanged


def foreground(x: torch.Tensor) -> torch.Tensor:
    """Bool [B, 1, H, W]: pixels of the ImageNet-normalised batch whose RGB mean is at least CELL_LUMINANCE_FLOOR."""
    return to_unit(x).mean(dim=1, keepdim=True) >= CELL_LUMINANCE_FLOOR


@torch.no_grad()
def apply_hed_jitter(x: torch.Tensor, fg: torch.Tensor, sigma: float = HED_SIGMA, prob: float = 1.0) -> torch.Tensor:
    """HED colour augmentation (Tellez et al., 2018 / 2019): per sample and HED channel s' = alpha * s + beta in the
    stain space of skimage's rgb2hed; the result is clipped to [0, 1] as hed2rgb does (no min-max rescale)."""
    b, device = x.shape[0], x.device
    selected = torch.rand(b, device=device) < prob
    alpha = 1.0 + (torch.rand(b, 3, 1, 1, device=device) * 2 - 1) * sigma
    beta = (torch.rand(b, 3, 1, 1, device=device) * 2 - 1) * sigma
    rgb = to_unit(x)
    hed_from_rgb, rgb_from_hed = _HED_FROM_RGB.to(rgb), _RGB_FROM_HED.to(rgb)
    hed = torch.einsum("ij,bihw->bjhw", hed_from_rgb, rgb.clamp(min=1e-6).log() / _HED_LOG_ADJUST).clamp(min=0.0)
    out = torch.einsum("ij,bihw->bjhw", rgb_from_hed, alpha * hed + beta).mul(_HED_LOG_ADJUST).exp().clamp(0.0, 1.0)
    return torch.where(selected.view(b, 1, 1, 1) & fg, from_unit(out).to(x.dtype), x)


@torch.no_grad()
def apply_randstainna(x: torch.Tensor, fg: torch.Tensor, stats: Dict, std_hyper: float = RANDSTAINNA_STD_HYPER,
                      prob: float = 1.0) -> torch.Tensor:
    """RandStainNA (Shen et al., MICCAI 2022), Lab: a virtual template per sample, mean ~ N(mean_avg, mean_std * (1 +
    std_hyper)) and sd ~ N(sd_avg, sd_std * (1 + std_hyper)) per channel from `fit_randstainna` stats, then Reinhard
    of the foreground to it. Template sds are floored at SD_FLOOR (the reference code leaves them unbounded)."""
    b, device = x.shape[0], x.device
    selected = torch.rand(b, device=device) < prob
    mean_avg, mean_std, sd_avg, sd_std = (torch.tensor(stats[k], device=device, dtype=torch.float32).view(1, 3, 1, 1)
                                          for k in RANDSTAINNA_KEYS)
    spread = 1.0 + std_hyper
    target_mean = mean_avg + mean_std * spread * torch.randn(b, 3, 1, 1, device=device)
    target_sd = (sd_avg + sd_std * spread * torch.randn(b, 3, 1, 1, device=device)).clamp(min=SD_FLOOR)
    lab = rgb_to_lab(to_unit(x))
    mean, sd, _ = region_moments(lab, (fg,))
    mean, sd = mean[:, 0, :, None, None], sd[:, 0, :, None, None].clamp(min=SD_FLOOR)
    out = lab_to_rgb((lab - mean) / sd * target_sd + target_mean)
    return torch.where(selected.view(b, 1, 1, 1) & fg, from_unit(out).to(x.dtype), x)


def fit_randstainna(train_dir: str, img_size: int = 224, batch: int = 64) -> Dict:
    """RandStainNA template distribution of the training set (the reference preprocess/datasets_statistics.py): the
    per-image foreground Lab mean and sd of every image under `train_dir/<class>/`, resized to `img_size` as the train
    transforms do, then the mean and population sd of each over the images. CPU, 3 threads, deterministic."""
    files = sorted(f for ext in ("png", "jpg", "jpeg", "bmp", "tif", "tiff")
                   for f in glob.glob(os.path.join(train_dir, "*", f"*.{ext}")))
    if not files:
        raise FileNotFoundError(f"no training image under {train_dir}/<class>/; run `python main.py prepare` first")
    threads = torch.get_num_threads()
    torch.set_num_threads(3)
    try:
        means, sds = [], []
        for start in range(0, len(files), batch):
            rgb = []
            for path in files[start:start + batch]:
                image = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                rgb.append(cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA))
            rgb = torch.from_numpy(np.stack(rgb)).permute(0, 3, 1, 2).double() / 255.0
            fg = rgb.mean(dim=1, keepdim=True) >= CELL_LUMINANCE_FLOOR
            mean, sd, _ = region_moments(rgb_to_lab(rgb), (fg,))
            means.append(mean[:, 0])
            sds.append(sd[:, 0])
    finally:
        torch.set_num_threads(threads)
    means, sds = torch.cat(means).numpy(), torch.cat(sds).numpy()
    return {
        "color_space": "CIE Lab (L 0-100), foreground = mean RGB >= CELL_LUMINANCE_FLOOR",
        "distribution": "normal",
        "n_images": len(files),
        "img_size": img_size,
        "mean_avg": means.mean(axis=0).tolist(),
        "mean_std": means.std(axis=0).tolist(),
        "sd_avg": sds.mean(axis=0).tolist(),
        "sd_std": sds.std(axis=0).tolist(),
    }


def rgb_to_od(rgb: torch.Tensor) -> torch.Tensor:
    """Optical density -ln(I / 255) of RGB in [0, 1], with I floored at 1 as in the Stain Mix-up code."""
    return -torch.log(rgb.clamp(min=1.0 / 255.0))


@torch.no_grad()
def macenko(od: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Two-stain matrices [B, 3, 2] of OD [B, 3, H, W] over bool `mask` [B, 1, H, W] (Macenko et al., 2009), and their
    validity [B] (at least MIN_STAIN_PX pixels with OD >= MACENKO_BETA in every channel). float32, except the 3x3
    eigen-decomposition (CPU, float64): consumer GPUs run float64 at 1/64 rate.

    Columns are unit OD vectors, the one with the larger red OD first (azure / methylene blue before eosin, as H before
    E in the reference ordering), so matrices of two images interpolate column by column. The angle extremes are the
    nearest-rank percentiles of a sort.
    """
    b = od.shape[0]
    flat = od.float().flatten(2)
    sel = mask.flatten(2) & (flat.amin(dim=1, keepdim=True) >= MACENKO_BETA)
    w = sel.float()
    n = w.sum(dim=-1)
    centred = (flat - ((flat * w).sum(dim=-1) / n.clamp(min=1)).unsqueeze(-1)) * w
    cov = centred @ centred.transpose(1, 2) / (n.unsqueeze(-1) - 1).clamp(min=1)
    _, vecs = torch.linalg.eigh(cov.double().cpu())
    plane = vecs[..., [2, 1]].to(flat)  # the two largest eigenvectors, [B, 3, 2]
    plane = plane * torch.where(plane.sum(dim=1, keepdim=True) < 0, -1.0, 1.0)
    proj = plane.transpose(1, 2) @ flat
    phi = torch.atan2(proj[:, 1], proj[:, 0])
    phi = torch.where(sel[:, 0], phi, torch.full_like(phi, float("inf"))).sort(dim=1).values
    last = (n - 1).clamp(min=0)
    ends = [phi.gather(1, (last * q / 100).round().long()) for q in (MACENKO_ALPHA, 100 - MACENKO_ALPHA)]
    stains = torch.stack([(plane @ torch.stack([e.cos(), e.sin()], dim=1)).squeeze(-1) for e in ends], dim=2)
    stains = torch.where((stains[:, 0, 0] < stains[:, 0, 1]).view(b, 1, 1), stains.flip(2), stains)
    stains = stains / stains.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return stains, n[:, 0] >= MIN_STAIN_PX


def estimate_stain_matrix(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Macenko stain matrix (3, 2) of one uint8 RGB image [H, W, 3] over bool `mask` [H, W] (float32; NaN when fewer
    than MIN_STAIN_PX usable pixels). The CPU path of `macenko`, deterministic."""
    unit = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].double() / 255.0
    stains, valid = macenko(rgb_to_od(unit), torch.from_numpy(np.ascontiguousarray(mask, dtype=bool))[None, None])
    return stains[0].numpy() if bool(valid[0]) else np.full((3, 2), np.nan, dtype=np.float32)


@torch.no_grad()
def apply_stain_mixup(x: torch.Tensor, fg: torch.Tensor, target_stains: torch.Tensor,
                      intensity: Tuple[float, float] = STAIN_INTENSITY, prob: float = 1.0) -> torch.Tensor:
    """Stain Mix-up (Chang et al., MICCAI 2021): S = a S_src + (1 - a) S_tgt with a ~ U[0, 1], columns renormalised,
    S_src the Macenko matrix of the sample's foreground and S_tgt a random row of `target_stains` [N, 3, 2] (the MLL23
    bank); the source concentrations, scaled per stain by U[intensity], are recomposed with S and no residual, as the
    reference code. Concentrations are non-negative least squares by clipping, not the reference's lasso (spams), so
    the arm stays deterministic. A sample without a valid source matrix is left unchanged."""
    b, device = x.shape[0], x.device
    selected = torch.rand(b, device=device) < prob
    a = torch.rand(b, 1, 1, device=device)
    target = target_stains.to(device)[torch.randint(len(target_stains), (b,), device=device)]
    intensity = intensity[0] + (intensity[1] - intensity[0]) * torch.rand(b, 2, 1, device=device)
    idx = selected.nonzero().squeeze(1)
    if idx.numel() == 0:
        return x
    rgb, fg_sel = to_unit(x[idx]), fg[idx]
    od = rgb_to_od(rgb).flatten(2)
    source, valid = macenko(rgb_to_od(rgb), fg_sel)
    mixed = a[idx] * source + (1 - a[idx]) * target[idx]
    mixed = mixed / mixed.norm(dim=1, keepdim=True).clamp(min=1e-8)
    gram = source.transpose(1, 2) @ source  # (S^T S)^-1 S^T od by the 2x2 adjugate: no solver kernel
    adj = torch.stack([gram[:, 1, 1], -gram[:, 0, 1], -gram[:, 1, 0], gram[:, 0, 0]], dim=1).view(-1, 2, 2)
    det = (gram[:, 0, 0] * gram[:, 1, 1] - gram[:, 0, 1] * gram[:, 1, 0]).clamp(min=1e-8).view(-1, 1, 1)
    conc = (adj / det @ source.transpose(1, 2) @ od).clamp(min=0.0) * intensity[idx]
    out = torch.exp(-(mixed @ conc)).view_as(rgb).clamp(0.0, 1.0)
    keep = valid.view(-1, 1, 1, 1) & fg_sel
    result = x.clone()
    result[idx] = torch.where(keep, from_unit(out).to(x.dtype), x[idx])
    return result
