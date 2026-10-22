"""MorphoMix: one Azure-B cell mask drives stain transfer (C1), small-cell rendering (C2'), background (C3) and
multi-cell field synthesis (C4)."""
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as nnf

from src.augmentations.transforms import IMAGENET_MEAN, IMAGENET_STD, resample_down_up
from src.augmentations.background import CELL_LUMINANCE_FLOOR, apply_background_randomization
from src.augmentations.stain import from_unit, match_regions, to_unit
from src.augmentations.stain_baselines import RANDSTAINNA_STD_HYPER, apply_randstainna, foreground
from src.augmentations.style_bank import StyleBank

# C1 acquisition simulation (`acquisition_prob`): light, so the cell stays a plausible in-focus capture
ACQ_BLUR = (0.0, 1.0)  # Gaussian sigma, px at 224
ACQ_NOISE = (0.0, 0.02)  # Gaussian sd, RGB in [0, 1]
ACQ_GAMMA = (0.8, 1.25)
ACQ_KERNEL = 7
# C2' / C4 rendering: anti-aliasing blur before a shrink (sigma up to 2 px, i.e. a shrink to 0.2x)
BLUR_MAX = 2.0
BLUR_TAPS = 13
MIN_CELL_PX = 64  # a smaller mask is no usable cell: never shrunk, never a C4 source
FIELD_SLOTS = 4  # C4 places its cells in distinct slots of a 4 x 4 grid (56 px at 224), so they never overlap


def binarize_heatmaps(heatmaps: torch.Tensor, threshold: float) -> torch.Tensor:
    """Hard masks M = 1[H >= tau] with the documented fallback rules."""
    masks = (heatmaps >= threshold).float()
    degenerate = heatmaps.amax(dim=(1, 2, 3), keepdim=True) < 1e-6
    empty = masks.sum(dim=(1, 2, 3), keepdim=True) == 0
    fallback = (heatmaps >= heatmaps.mean(dim=(1, 2, 3), keepdim=True) * 0.8).float()
    masks = torch.where(empty, fallback, masks)
    return torch.where(degenerate, torch.zeros_like(masks), masks)


def transfer_appearance(images: torch.Tensor, masks: torch.Tensor, bank: StyleBank, prob: float,
                        alpha: Tuple[float, float], virtual_prob: float,
                        exclude: Optional[torch.Tensor] = None) -> torch.Tensor:
    """C1: per-region Lab Reinhard of the cell towards a reference drawn per sample (a random MLL23 bank cell, or with
    `virtual_prob` a virtual template), blended with strength a ~ U[alpha] (Stain Mix-up-style interpolation between
    the own and the reference stain). Pixels outside the mask and samples in `exclude` are unchanged."""
    b, device = images.shape[0], images.device
    selected = torch.rand(b, device=device) < prob
    if exclude is not None:
        selected = selected & ~exclude
    lo, hi = alpha
    strength = lo + (hi - lo) * torch.rand(b, 1, 1, 1, device=device)
    ref_mean, ref_sd = bank.draw(b, virtual_prob, device)
    idx = selected.nonzero().squeeze(1)
    if idx.numel() == 0:
        return images
    rgb = to_unit(images[idx])
    matched = match_regions(rgb, masks[idx], ref_mean[idx], ref_sd[idx])
    out = images.clone()
    out[idx] = torch.where(masks[idx] > 0.5, from_unit(rgb + strength[idx] * (matched - rgb)), images[idx])
    return out


def gaussian_blur(x: torch.Tensor, sigma: torch.Tensor, size: int) -> torch.Tensor:
    """Separable Gaussian blur of [n, c, H, W] with a per-sample sigma (n values, px) and `size` taps, replicate
    padding; sigma ~ 0 is the identity."""
    n, c, h, w = x.shape
    taps = torch.arange(size, device=x.device, dtype=x.dtype) - size // 2
    kernel = torch.exp(-0.5 * (taps / sigma.view(n, 1).clamp(min=1e-3)) ** 2)
    kernel = (kernel / kernel.sum(dim=1, keepdim=True)).repeat_interleave(c, dim=0)
    pad = size // 2
    flat = x.reshape(1, n * c, h, w)
    flat = nnf.conv2d(nnf.pad(flat, (pad, pad, 0, 0), mode="replicate"), kernel.view(n * c, 1, 1, -1), groups=n * c)
    flat = nnf.conv2d(nnf.pad(flat, (0, 0, pad, pad), mode="replicate"), kernel.view(n * c, 1, -1, 1), groups=n * c)
    return flat.view(n, c, h, w)


def simulate_acquisition(images: torch.Tensor, masks: torch.Tensor, prob: float) -> torch.Tensor:
    """Optional part of C1: another microscope and camera inside the cell, per sample defocus blur (Gaussian, sigma
    ~ U[ACQ_BLUR]), sensor noise (Gaussian, sd ~ U[ACQ_NOISE]) and a tone curve (gamma, log-uniform in ACQ_GAMMA)."""
    b, device = images.shape[0], images.device
    selected = torch.rand(b, device=device) < prob
    sigma = ACQ_BLUR[0] + (ACQ_BLUR[1] - ACQ_BLUR[0]) * torch.rand(b, 1, device=device)
    noise_sd = ACQ_NOISE[0] + (ACQ_NOISE[1] - ACQ_NOISE[0]) * torch.rand(b, 1, 1, 1, device=device)
    log_gamma = torch.tensor(ACQ_GAMMA, device=device).log()
    gamma = (log_gamma[0] + (log_gamma[1] - log_gamma[0]) * torch.rand(b, 1, 1, 1, device=device)).exp()
    noise = torch.randn(images.shape, device=device)
    idx = selected.nonzero().squeeze(1)
    if idx.numel() == 0:
        return images
    rgb = to_unit(images[idx])
    blurred = gaussian_blur(rgb, sigma[idx], ACQ_KERNEL)
    acquired = (blurred + noise_sd[idx] * noise[idx]).clamp(0.0, 1.0) ** gamma[idx]
    out = images.clone()
    out[idx] = torch.where(masks[idx] > 0.5, from_unit(acquired), images[idx])
    return out


def cell_geometry(masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per sample: the mask centroid [B, 2] (x, y in normalised [-1, 1] coordinates), its equivalent diameter [B]
    (px) and whether it is a usable cell (>= MIN_CELL_PX pixels)."""
    _, _, h, w = masks.shape
    m = masks[:, 0].float()
    area = m.sum(dim=(1, 2))
    ys = (torch.arange(h, device=m.device, dtype=m.dtype) + 0.5) / h * 2 - 1
    xs = (torch.arange(w, device=m.device, dtype=m.dtype) + 0.5) / w * 2 - 1
    cy = (m.sum(dim=2) * ys).sum(dim=1) / area.clamp(min=1)
    cx = (m.sum(dim=1) * xs).sum(dim=1) / area.clamp(min=1)
    return torch.stack([cx, cy], dim=1), 2 * (area / math.pi).sqrt(), area >= MIN_CELL_PX


def render_cells(rgb: torch.Tensor, masks: torch.Tensor, src: torch.Tensor, scale: torch.Tensor, angle: torch.Tensor,
                 flip: torch.Tensor, centre: torch.Tensor, origin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The cells of rows `src` of `rgb` ([B, 3, H, W] in [0, 1]) and `masks`, each zoomed by `scale` (< 1 shrinks),
    rotated by `angle`, mirrored where `flip` is -1 and moved from its centroid `origin` to `centre` (normalised x, y).
    A shrink is anti-aliased by a Gaussian of sigma (1 / s - 1) / 2 px on the premultiplied cell (normalised
    convolution), so the black surround never darkens the rim. Returns the cell colour [n, 3, H, W] (0 outside the
    cell) and its coverage alpha [n, 1, H, W]."""
    m = masks[src].to(rgb.dtype)
    sigma = ((1.0 / scale - 1.0) / 2).clamp(0.0, BLUR_MAX)
    pre = gaussian_blur(torch.cat([rgb[src] * m, m], dim=1), sigma, BLUR_TAPS)
    cos, sin = torch.cos(angle), torch.sin(angle)
    a = torch.stack([torch.stack([cos * flip, -sin], dim=1), torch.stack([sin * flip, cos], dim=1)], dim=1)
    a = a / scale.view(-1, 1, 1)  # output -> source: q = A (p - centre) + origin
    theta = torch.cat([a, (origin - (a @ centre.unsqueeze(2)).squeeze(2)).unsqueeze(2)], dim=2)
    grid = nnf.affine_grid(theta, list(pre.shape), align_corners=False)
    out = nnf.grid_sample(pre, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    alpha = out[:, 3:]
    colour = torch.where(alpha > 1e-4, out[:, :3] / alpha.clamp(min=1e-4), torch.zeros_like(out[:, :3]))
    return colour.clamp(0.0, 1.0), alpha


def small_cells(images: torch.Tensor, masks: torch.Tensor, prob: float, size_px: Tuple[float, float],
                lowdetail_prob: float, exclude: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """C2' small cell: with `prob` (samples outside `exclude` only), the cell is shrunk about its centroid to an
    equivalent diameter of t ~ U[size_px] px, the size of a white cell in a whole field at img_size (Aria: ~27 px; a
    C-NMC cell: ~100 px); a cell is never enlarged. With `lowdetail_prob` it is zoomed back to its own size
    (`resample_down_up` of the whole crop: the detail of a field cell cropped and resized to the training scale, what
    cell-by-cell field inference sees), else it stays small at a random position (the whole-field scale). Returns the
    images, the masks and the samples that stay small, which C3 always gives a field background."""
    b, device = images.shape[0], images.device
    selected = torch.rand(b, device=device) < prob
    target = size_px[0] + (size_px[1] - size_px[0]) * torch.rand(b, device=device)
    lowdetail = torch.rand(b, device=device) < lowdetail_prob
    pos = torch.rand(b, 2, device=device) * 2 - 1
    origin, diameter, valid = cell_geometry(masks)
    selected = selected & valid & ~exclude
    small = selected & ~lowdetail
    s = (target / diameter.clamp(min=1.0)).clamp(max=1.0)
    out_images, out_masks = images, masks
    if bool((selected & lowdetail).any()):
        out_images = resample_down_up(images, selected & lowdetail, s)
    idx = small.nonzero().squeeze(1)
    if idx.numel() > 0:
        margin = 1.0 - 1.2 * target[idx] / images.shape[2]  # the cell's half span (normalised) plus 20 % stays inside
        rgb, alpha = render_cells(to_unit(images), masks, idx, s[idx], torch.zeros_like(s[idx]),
                                  torch.ones_like(s[idx]), pos[idx] * margin.view(-1, 1), origin[idx])
        hard = (alpha > 0.5).to(masks.dtype)
        out_images, out_masks = out_images.clone(), masks.clone()
        out_images[idx] = from_unit(rgb * hard)
        out_masks[idx] = hard
    return out_images, out_masks, small


def synthesize_fields(images: torch.Tensor, masks: torch.Tensor, labels: torch.Tensor, prob: float,
                      size_px: Tuple[float, float], n_cells: Tuple[int, int]
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """C4 multi-cell field: with `prob`, a sample becomes a field of n ~ U{n_cells} cells of its own class: its own
    cell plus cells of batch samples with the same label (drawn with replacement, so a class with one sample in the
    batch gives differently transformed copies of the own cell). Each is shrunk to t ~ U[size_px] px, rotated,
    mirrored with p 0.5 and placed in its own slot of a FIELD_SLOTS x FIELD_SLOTS grid, jittered but inside the slot,
    so cells never overlap. The label is unchanged. Returns the images, the masks (union of the placed cells) and the
    field samples, which C3 always gives a field background. Every draw has a fixed shape."""
    b, device = images.shape[0], images.device
    k_max = int(n_cells[1])
    if not 1 <= n_cells[0] <= k_max <= FIELD_SLOTS ** 2:
        raise ValueError(f"field_cells {n_cells}: need 1 <= min <= max <= {FIELD_SLOTS ** 2}")
    selected = torch.rand(b, device=device) < prob
    count = n_cells[0] + (torch.rand(b, device=device) * (k_max - n_cells[0] + 1)).long().clamp(max=k_max - n_cells[0])
    pick = torch.rand(b, k_max, device=device)
    target = size_px[0] + (size_px[1] - size_px[0]) * torch.rand(b, k_max, device=device)
    angle = torch.rand(b, k_max, device=device) * (2 * math.pi)
    flip = torch.where(torch.rand(b, k_max, device=device) < 0.5, -1.0, 1.0)
    slots = torch.rand(b, FIELD_SLOTS ** 2, device=device).argsort(dim=1)[:, :k_max]
    jitter = torch.rand(b, k_max, 2, device=device) * 2 - 1

    origin, diameter, valid = cell_geometry(masks)
    selected = selected & valid
    idx = selected.nonzero().squeeze(1)
    if idx.numel() == 0:
        return images, masks, selected
    # source of slot k: the (r + 1)-th usable sample of the same label, r = floor(pick * their count); slot 0 = own cell
    same = (labels.view(-1, 1) == labels.view(1, -1)) & valid.view(1, -1)
    available = same.sum(dim=1, keepdim=True)
    rank = (pick * available).long().minimum((available - 1).clamp(min=0))
    hit = (same.long().cumsum(dim=1).unsqueeze(1) == (rank + 1).unsqueeze(2)) & same.unsqueeze(1)
    src = hit.long().argmax(dim=2)
    src[:, 0] = torch.arange(b, device=device)

    rgb = to_unit(images)
    half_slot = 1.0 / FIELD_SLOTS
    canvas = torch.zeros(idx.numel(), 3, *images.shape[2:], device=device)
    field_mask = torch.zeros(idx.numel(), 1, *images.shape[2:], device=device, dtype=masks.dtype)
    for k in range(k_max):
        active = count[idx] > k
        if not bool(active.any()):
            break
        src_k = src[idx, k]
        s = (target[idx, k] / diameter[src_k].clamp(min=1.0)).clamp(max=1.0)
        slot = slots[idx, k]
        centre = torch.stack([slot % FIELD_SLOTS, slot // FIELD_SLOTS], dim=1).float() * (2 * half_slot) + (
            half_slot - 1.0)
        reach = (half_slot - 1.15 * target[idx, k] / images.shape[2]).clamp(min=0.0)  # half span + 15 % stays inside
        colour, alpha = render_cells(rgb, masks, src_k, s, angle[idx, k], flip[idx, k],
                                     centre + jitter[idx, k] * reach.view(-1, 1), origin[src_k])
        hard = (alpha > 0.5) & active.view(-1, 1, 1, 1)
        canvas = torch.where(hard, colour, canvas)
        field_mask = torch.maximum(field_mask, hard.to(masks.dtype))
    out_images, out_masks = images.clone(), masks.clone()
    out_images[idx] = from_unit(canvas)
    out_masks[idx] = field_mask
    return out_images, out_masks, selected


def apply_morpho_mix(
    images: torch.Tensor,
    cam_heatmaps: torch.Tensor,
    threshold: float = 0.5,
    bank: Optional[StyleBank] = None,
    appearance_prob: float = 0.0,
    appearance_alpha: Tuple[float, float] = (0.5, 1.0),
    virtual_template_prob: float = 0.0,
    acquisition_prob: float = 0.0,
    small_cell_prob: float = 0.0,
    small_cell_px: Tuple[float, float] = (24.0, 48.0),
    small_cell_lowdetail_prob: float = 0.5,
    field_prob: float = 0.0,
    field_cells: Tuple[int, int] = (2, 8),
    labels: Optional[torch.Tensor] = None,
    use_background: bool = True,
    background_prob: float = 0.5,
    background_rbc: Tuple[int, int] = (5, 25),
    field_rbc: Tuple[int, int] = (40, 120),
    feather_edges: bool = True,
    rsn_prob: float = 0.0,
    rsn_stats: Optional[Dict] = None,
    rsn_std_hyper: float = RANDSTAINNA_STD_HYPER,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MorphoMix on a normalised batch; returns the augmented images and their cell masks. Labels are unchanged.

    The order matters: C1 first, on the dataset transforms' output (flips, rotation, mild colour jitter), so the
    region statistics are those of the cell on its own crop. C4 then builds its fields from the C1 output and C2'
    shrinks the cells of the remaining samples (a sample gets at most one of them). C3 runs last, so the area that
    C2' and C4 leave empty becomes a (dense, field-scale) smear; with C3 off it stays black, as in C-NMC. `bank` None
    turns C1 off; C4 needs the batch `labels`.

    With `rsn_prob` > 0 the stain stage is a hybrid: per sample, with `rsn_prob`, RandStainNA (`apply_randstainna`,
    the training-set template distribution `rsn_stats`) recolours the foreground, exactly as the randstainna arm. The
    C1 and RandStainNA draws are independent and RandStainNA takes precedence, so a sample gets RandStainNA (p
    rsn_prob), C1 (p appearance_prob * (1 - rsn_prob)) or neither. `acquisition_prob` stays independent of both.
    """
    masks = binarize_heatmaps(cam_heatmaps, threshold)
    # clamped to cell pixels of the original image
    lum = (images * torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
           + torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)).mean(dim=1, keepdim=True)
    clamped = masks * (lum >= CELL_LUMINANCE_FLOOR).to(masks.dtype)
    # very dark cell: keep the unclamped mask
    masks = torch.where((clamped.flatten(1).sum(dim=1) > 0).view(-1, 1, 1, 1), clamped, masks)

    rsn = None  # None leaves C1 exactly as without the hybrid
    if rsn_prob > 0:
        if rsn_stats is None:
            raise ValueError("rsn_prob > 0 needs the RandStainNA stats (randstainna_stats)")
        rsn = torch.rand(images.shape[0], device=images.device) < rsn_prob
        # prob 1 inside: the selection is `rsn`, applied through the foreground
        images = apply_randstainna(images, foreground(images) & rsn.view(-1, 1, 1, 1), rsn_stats,
                                   std_hyper=rsn_std_hyper, prob=1.0)
    if bank is not None:
        images = transfer_appearance(images, masks, bank, appearance_prob, appearance_alpha, virtual_template_prob,
                                     exclude=rsn)
    if acquisition_prob > 0:
        images = simulate_acquisition(images, masks, acquisition_prob)
    fields = None  # samples at field scale; None leaves C3 exactly as without C2' and C4
    if field_prob > 0 or small_cell_prob > 0:
        c4 = small = torch.zeros(images.shape[0], dtype=torch.bool, device=images.device)
        if field_prob > 0:
            if labels is None:
                raise ValueError("C4 (field_prob > 0) needs the batch labels")
            field_images, field_masks, c4 = synthesize_fields(images, masks, labels, field_prob, small_cell_px,
                                                              field_cells)
        if small_cell_prob > 0:
            images, masks, small = small_cells(images, masks, small_cell_prob, small_cell_px,
                                               small_cell_lowdetail_prob, exclude=c4)
        if field_prob > 0:
            images = torch.where(c4.view(-1, 1, 1, 1), field_images, images)
            masks = torch.where(c4.view(-1, 1, 1, 1), field_masks, masks)
        fields = c4 | small
    if use_background:
        images = apply_background_randomization(images, masks, prob=background_prob,
                                                feather_kernel=5 if feather_edges else 1, n_rbc=background_rbc,
                                                fields=fields, field_rbc=field_rbc)
    return images, masks
