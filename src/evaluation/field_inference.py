"""Whole-field inference at the training cell scale: detect the white cells of a smear field, crop each one with a
window that gives it the size of a training cell, score every crop and aggregate the crop scores to a field score.

Aria fields are 224 px JPEGs with white cells of ~26 px (median equivalent diameter), while the C-NMC training cells
are ~95 px at 224 px. Resizing the whole field therefore shows the classifier cells four times smaller than any it
was trained on; `cells` inference instead crops a window of 224 / `scale` field px around each detected cell (scale
3.6 brings a cell to the training size; the config uses 4.5, chosen on the tests).
"""
from typing import Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.augmentations.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.cam.blast_prior import extract_blast_cell_prior
from src.evaluation.tta import tta_logits

AGGREGATES = ("mean", "max", "topk", "frac")


def white_cell_mask(rgb: np.ndarray) -> np.ndarray:
    """{0, 1} uint8 mask of the white cells of a smear field: the saturated mode of the field (Otsu on HSV saturation
    once to drop the background, once more among the remaining pixels to drop the paler erythrocytes) united with the
    Azure-B prior mask. The prior alone misses the dark blue lymphocytes of Aria's Benign fields (hue below its
    purple range), saturation alone misses pale purple blasts."""
    sat = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1], (5, 5), 1.0)
    t_background, _ = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    stained = sat[sat > t_background]
    t_cells = cv2.threshold(stained.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] if stained.size \
        else t_background
    mask = (sat > t_cells).astype(np.uint8) | extract_blast_cell_prior(rgb)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)


def detect_cells(rgb: np.ndarray, min_radius: float, separation: float = 0.8, min_inside: float = 0.0,
                 min_nucleus: float = 0.0) -> np.ndarray:
    """Cell centres and radii, float32 [N, 3] (x, y, r) in field px, largest first.

    Centres are the maxima of the distance transform of `white_cell_mask` (r = the distance to the mask edge), so
    touching cells give one centre each; a maximum closer than `separation` x (r1 + r2) to a larger one is dropped,
    and so is one with r < `min_radius` (platelets, erythrocyte fragments). Optional whole-cell filter: a cell is
    also dropped when less than `min_inside` of its disk at 1.5 r (the cell and its rim) lies inside the field (cut
    by the border) or less than `min_nucleus` of its disk is Azure-B prior inside the mask (no nucleus: an
    erythrocyte or pale fragment). 0 keeps every cell; a field the filter would empty keeps all its cells (the
    prior misses the dark blue lymphocytes of some Benign fields), so it is still scored cell by cell.
    """
    mask = white_cell_mask(rgb)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    local_max = dist >= cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) - 1e-3
    n, _, _, centroids = cv2.connectedComponentsWithStats(((dist >= min_radius) & local_max).astype(np.uint8))
    candidates = sorted(((float(dist[int(round(y)), int(round(x))]), float(x), float(y)) for x, y in centroids[1:]),
                        reverse=True)
    kept: List[tuple] = []
    for r, x, y in candidates:
        if all(np.hypot(x - kx, y - ky) > separation * (r + kr) for kr, kx, ky in kept):
            kept.append((r, x, y))
    if min_inside > 0 or min_nucleus > 0:
        nucleus = extract_blast_cell_prior(rgb)[1].astype(bool) & mask.astype(bool)
        yy, xx = np.mgrid[:rgb.shape[0], :rgb.shape[1]]
        whole = []
        for r, x, y in kept:
            d2 = (xx - x) ** 2 + (yy - y) ** 2
            inside = (d2 <= (1.5 * r) ** 2).sum() / (np.pi * (1.5 * r) ** 2)
            whole.append(inside >= min_inside and nucleus[d2 <= r * r].mean() >= min_nucleus)
        if any(whole):
            kept = [k for k, w in zip(kept, whole) if w]
    return np.asarray([(x, y, r) for r, x, y in kept], dtype=np.float32).reshape(-1, 3)


def crop_cells(fields: torch.Tensor, owner: torch.Tensor, centres: torch.Tensor, window: float,
               out_size: int) -> torch.Tensor:
    """Square crops [N, 3, out, out] of side `window` field px centred on `centres` [N, 2] (x, y in px) of the
    fields [B, 3, H, W] (values in [0, 1]) indexed by `owner` [N]; bicubic, edge-replicated outside the field."""
    h, w = fields.shape[-2:]
    theta = torch.zeros(len(owner), 2, 3, device=fields.device, dtype=fields.dtype)
    theta[:, 0, 0] = window / w
    theta[:, 1, 1] = window / h
    theta[:, 0, 2] = (2 * centres[:, 0] + 1) / w - 1  # pixel centres at (2i + 1) / W - 1, align_corners=False
    theta[:, 1, 2] = (2 * centres[:, 1] + 1) / h - 1
    grid = F.affine_grid(theta, [len(owner), 3, out_size, out_size], align_corners=False)
    return F.grid_sample(fields[owner], grid, mode="bicubic", padding_mode="border", align_corners=False).clamp(0, 1)


def aggregate(p: np.ndarray, how: str, top_k: int = 3, threshold: float = 0.5) -> float:
    """One field score from its crop scores p(ALL): mean, max, mean of the `top_k` highest, or the fraction at
    or above `threshold`."""
    if how == "mean":
        return float(p.mean())
    if how == "max":
        return float(p.max())
    if how == "topk":
        return float(np.sort(p)[-top_k:].mean())
    if how == "frac":
        return float((p >= threshold).mean())
    raise ValueError(f"aggregate must be one of {AGGREGATES}, got {how!r}")


def crop_batch(fields_u8: Sequence[np.ndarray], jobs: Sequence[tuple], scale: float, out_size: int,
               device: torch.device) -> torch.Tensor:
    """ImageNet-normalised crops [N, 3, out, out] of the (field index, `detect_cells` row) `jobs`, window out_size /
    `scale` field px."""
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    ids = sorted({i for i, _ in jobs})
    local = {i: k for k, i in enumerate(ids)}
    fields = torch.from_numpy(np.stack([fields_u8[i] for i in ids])).to(device).permute(0, 3, 1, 2).float() / 255
    owner = torch.tensor([local[i] for i, _ in jobs], device=device)
    centres = torch.tensor(np.stack([c[:2] for _, c in jobs]), device=device, dtype=torch.float32)
    return (crop_cells(fields, owner, centres, out_size / scale, out_size) - mean) / std


@torch.no_grad()
def score_crops(model: torch.nn.Module, fields_u8: Sequence[np.ndarray], cells: Sequence[np.ndarray], scale: float,
                out_size: int, tta_views: int, device: torch.device, batch_size: int = 128,
                autocast: Optional[Callable] = None, channels_last: bool = False) -> List[np.ndarray]:
    """p(ALL) of every detected cell of every field, one float64 array per field (empty when none was found).

    Args:
        fields_u8: uint8 RGB fields [H, W, 3].
        cells: `detect_cells` output per field.
        scale: crop window = out_size / scale field px.
        autocast: context-manager factory for the forward pass (the Trainer's accelerator.autocast).
    """
    jobs = [(i, c) for i, cs in enumerate(cells) for c in cs]
    out: List[List[float]] = [[] for _ in fields_u8]
    for s in range(0, len(jobs), batch_size):
        chunk = jobs[s:s + batch_size]
        x = crop_batch(fields_u8, chunk, scale, out_size, device)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with autocast() if autocast is not None else torch.autocast(device.type, enabled=False):
            p = tta_logits(model, x, n_views=tta_views).float().exp()[:, 1].cpu().numpy()
        for (i, _), v in zip(chunk, p):
            out[i].append(float(v))
    return [np.asarray(v, dtype=np.float64) for v in out]


def field_scores(cell_probs: Sequence[np.ndarray], fallback: np.ndarray, how: str, top_k: int = 3,
                 threshold: float = 0.5) -> np.ndarray:
    """Field p(ALL) [N] from the crop scores; a field without a detected cell keeps its `fallback` (whole-field
    resize) score."""
    return np.asarray([aggregate(p, how, top_k, threshold) if p.size else float(f)
                       for p, f in zip(cell_probs, fallback)], dtype=np.float64)


def crop_scales(settings: Dict) -> List[float]:
    """`cells.scale`: one number, or a list whose crop scores `predict_fields` averages per cell."""
    scale = settings["scale"]
    return [float(s) for s in scale] if isinstance(scale, (list, tuple)) else [float(scale)]


def settings_summary(settings: Dict) -> str:
    whole = (f", whole cells (inside >= {settings['min_inside']}, nucleus >= {settings['min_nucleus']})"
             if settings.get("min_inside", 0) or settings.get("min_nucleus", 0) else "")
    return (f"cells: scale {settings['scale']}, min radius {settings['min_radius']} px{whole}, aggregate "
            f"{settings['aggregate']}" + (f" (k {settings['top_k']})" if settings["aggregate"] == "topk" else ""))
