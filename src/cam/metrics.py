"""XAI metrics (average drop, increase in confidence, pointing game, energy in mask, IoU) and the pseudo ground
truth they use."""
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.augmentations.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.datasets.cells import cell_mask, centre_cell


def compute_average_drop(
    orig_prob: float,
    masked_prob: float
) -> float:
    """Computes Average Drop (%) for a single prediction."""
    if orig_prob <= 1e-8:
        return 0.0
    return float(max(0.0, (orig_prob - masked_prob) / orig_prob) * 100.0)


def compute_pointing_game(
    cam_heatmap: np.ndarray,
    cell_mask: np.ndarray
) -> bool:
    """True when the CAM peak falls inside the cell mask."""
    if cam_heatmap.max() == 0:
        return False
    peak_y, peak_x = np.unravel_index(np.argmax(cam_heatmap), cam_heatmap.shape)
    return bool(cell_mask[peak_y, peak_x] > 0)


def compute_energy_in_mask(
    cam_heatmap: np.ndarray,
    cell_mask: np.ndarray
) -> float:
    """Share (%) of CAM energy inside the cell mask."""
    total_energy = float(np.sum(cam_heatmap))
    if total_energy < 1e-8:
        return 0.0
    binary_mask = (cell_mask > 0).astype(np.float32)
    in_mask_energy = float(np.sum(cam_heatmap * binary_mask))
    return float(min(100.0, (in_mask_energy / total_energy) * 100.0))


def compute_cam_iou(
    cam_heatmap: np.ndarray,
    cell_mask: np.ndarray,
    threshold: float = 0.50
) -> float:
    """Computes Intersection-over-Union."""
    cam_bin = (cam_heatmap >= threshold).astype(bool)
    mask_bin = (cell_mask > 0).astype(bool)

    intersection = np.logical_and(cam_bin, mask_bin).sum()
    union = np.logical_or(cam_bin, mask_bin).sum()

    if union == 0:
        return 100.0 if not cam_bin.any() and not mask_bin.any() else 0.0
    return float((intersection / union) * 100.0)


@torch.no_grad()
def evaluate_xai_metrics(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    cam_maps: torch.Tensor,
    cell_masks: torch.Tensor,
    device: torch.device = torch.device("cpu")
) -> Dict[str, float]:
    """Batch means; drop/increase perturb the image, the localisation metrics use `cell_masks`."""
    model.eval()
    images = images.to(device)
    targets = targets.to(device)

    if cam_maps.ndim == 3:
        cam_maps = cam_maps.unsqueeze(1)
    cam_maps = cam_maps.to(device)

    B, C, H, W = images.shape

    orig_logits = model(images)
    orig_probs = F.softmax(orig_logits, dim=-1)

    mean = torch.tensor(IMAGENET_MEAN, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    denormed = images * std + mean

    # mask in reflectance space, then renormalise
    masked_pixel = denormed * cam_maps

    masked_images = (masked_pixel - mean) / std
    masked_logits = model(masked_images)
    masked_probs = F.softmax(masked_logits, dim=-1)

    drops = []
    increases = 0
    pointing_hits = 0
    energy_ratios = []
    ious = []

    cam_np = cam_maps.squeeze(1).cpu().numpy()
    mask_np = cell_masks.cpu().numpy()

    for b in range(B):
        c = targets[b].item()
        p_orig = orig_probs[b, c].item()
        p_masked = masked_probs[b, c].item()

        drop = compute_average_drop(p_orig, p_masked)
        drops.append(drop)

        if p_masked > p_orig:
            increases += 1

        hit = compute_pointing_game(cam_np[b], mask_np[b])
        if hit:
            pointing_hits += 1

        energy = compute_energy_in_mask(cam_np[b], mask_np[b])
        energy_ratios.append(energy)

        iou = compute_cam_iou(cam_np[b], mask_np[b], threshold=0.45)
        ious.append(iou)

    return {
        "avg_drop_pct": float(np.mean(drops)),
        "increase_conf_pct": float((increases / B) * 100.0),
        "pointing_game_hit_rate_pct": float((pointing_hits / B) * 100.0),
        "energy_in_mask_pct": float(np.mean(energy_ratios)),
        "mean_iou_pct": float(np.mean(ious))
    }


def central_cell_mask(rgb: np.ndarray, threshold: float) -> np.ndarray:
    """XAI pseudo ground truth, {0, 1} float32 [H, W]: the Azure-B cell mask at `threshold` (the mask MorphoMix
    trains on) cut to the cell at the image centre, since on a smear background it also marks erythrocytes."""
    return centre_cell(cell_mask(np.ascontiguousarray(rgb), threshold))
