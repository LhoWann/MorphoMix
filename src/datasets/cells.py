"""Single-cell image IO and the cell masks: the Azure-B masks of C1's own cells, the MLL23 style bank and the XAI
pseudo ground truth, and `smear_cell` for the auxiliary training cells."""
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image

from src.augmentations.background import CELL_LUMINANCE_FLOOR
from src.cam.blast_prior import extract_blast_cell_prior

IMAGE_SUFFIXES = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")
CENTRE_OPEN_PX = 9  # opening that cuts the centre cell free of touching erythrocytes
NUCLEUS_OPEN_PX = 5  # opening of the MLL23 nucleus seed and of the grown cell


def list_images(root: str) -> List[str]:
    """Sorted image paths under root, skipping macOS resource forks."""
    return sorted(
        str(p) for p in Path(root).rglob("*")
        if p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith("._") and "__MACOSX" not in p.parts
    )


def load_rgb(path: str, img_size: int) -> np.ndarray:
    """uint8 RGB [img_size, img_size, 3]: shorter side resized to img_size, then center-cropped."""
    with Image.open(path) as im:
        rgb = np.array(im.convert("RGB"))
    h, w = rgb.shape[:2]
    scale = img_size / min(h, w)
    if scale != 1.0:
        size = (max(img_size, round(w * scale)), max(img_size, round(h * scale)))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
    h, w = rgb.shape[:2]
    top, left = (h - img_size) // 2, (w - img_size) // 2
    return np.ascontiguousarray(rgb[top:top + img_size, left:left + img_size])


def cell_mask(rgb: np.ndarray, threshold: float) -> np.ndarray:
    """{0, 1} float32 [H, W] mask; the same as `apply_morpho_mix` builds: binarize_heatmaps(PriorOnlyCAM.generate(x),
    threshold), clamped to pixels of mean RGB >= CELL_LUMINANCE_FLOOR unless that empties it (a very dark cell).

    The clamp removes the dilated prior's rim on the black C-NMC crop background (~14% of the unclamped mask); on
    smear images it is a no-op. After per-image min-max normalisation the maximum is 1, so the empty-mask fallback of
    binarize_heatmaps never fires and only the all-zero (degenerate) prior yields an empty mask.
    """
    soft = extract_blast_cell_prior(rgb)[0]
    lo, hi = float(soft.min()), float(soft.max())
    if hi < 1e-6:
        return np.zeros_like(soft, dtype=np.float32)
    mask = (soft - lo) / max(hi - lo, 1e-8) >= threshold
    clamped = mask & (rgb.mean(axis=-1) >= CELL_LUMINANCE_FLOOR * 255.0)
    return (clamped if clamped.any() else mask).astype(np.float32)


def centre_cell(mask: np.ndarray, open_px: int = CENTRE_OPEN_PX) -> np.ndarray:
    """The cell at the image centre only: MLL23 crops centre one white cell, but the Azure-B prior also passes
    the lilac erythrocytes around it. An opening cuts thin bridges, the component nearest the centre is kept, and
    a geodesic dilation inside the original mask restores its edge."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    n, labels, _, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if n <= 1:
        return mask
    h, w = mask.shape
    at_centre = labels[h // 2, w // 2]
    if at_centre == 0:
        at_centre = 1 + int(np.argmin(np.hypot(centroids[1:, 0] - w / 2, centroids[1:, 1] - h / 2)))
    cell = (labels == at_centre).astype(np.uint8)
    for _ in range(open_px):
        cell = cv2.dilate(cell, kernel) & mask.astype(np.uint8)
    return cell.astype(np.float32)


def nucleus_cell(rgb: np.ndarray, mask: np.ndarray, ring: int, min_sat: int,
                 open_px: int = NUCLEUS_OPEN_PX) -> np.ndarray:
    """The white cell at the image centre as its nucleus plus a cytoplasm ring, for MLL23, whose prior mask also
    covers the lilac erythrocytes (median 0.59 of the frame vs 0.20 on C-NMC).

    Inside the prior mask, saturation is bimodal: erythrocytes ~40-55, nuclei ~170-190; Otsu splits them and the
    opened component nearest the centre is the nucleus. It then grows by `ring` px into pixels with HSV saturation
    above `min_sat`, which takes the blue cytoplasm (~65-110) but stops at the paler erythrocytes and background.
    """
    sat = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1]
    inside = mask > 0
    if not inside.any():
        return mask
    otsu, _ = cv2.threshold(sat[inside].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
    strong = cv2.morphologyEx(((sat > otsu) & inside).astype(np.uint8), cv2.MORPH_OPEN, kernel)
    n, labels, _, centroids = cv2.connectedComponentsWithStats(strong, connectivity=8)
    if n <= 1:
        return centre_cell(mask)
    h, w = mask.shape
    at_centre = labels[h // 2, w // 2]
    if at_centre == 0:
        at_centre = 1 + int(np.argmin(np.hypot(centroids[1:, 0] - w / 2, centroids[1:, 1] - h / 2)))
    cell = (labels == at_centre).astype(np.uint8)
    stained = (sat > min_sat).astype(np.uint8)
    step = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(ring):
        cell = cv2.dilate(cell, step) & stained
    cell = cv2.morphologyEx(cell, cv2.MORPH_OPEN, kernel)
    outside = np.pad(cell, 1)  # the zero frame guarantees a background seed for the hole filling
    cv2.floodFill(outside, None, (0, 0), 1)
    return (cell | (1 - outside[1:-1, 1:-1])).astype(np.float32)


def smear_cell(rgb: np.ndarray, rim_px: int = 7) -> np.ndarray:
    """{0, 1} uint8 mask of the white cell at the centre of a smear crop whose erythrocytes and background are warm
    (orange-pink, as in May-Grunwald-Giemsa smears), for auxiliary training cells.

    Nucleus seed: Otsu on HSV saturation restricted to violet hues, every component near the centre (all lobes of a
    segmented nucleus). GrabCut then grows it (nucleus = sure foreground, frame border = sure background); grown pixels
    of warm hue are dropped beyond `rim_px` of the nucleus, so a touching erythrocyte is not taken along while the
    pale cytoplasm rim stays, the same for every class. Empty when no nucleus is found.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat = hsv[..., 0], cv2.GaussianBlur(hsv[..., 1], (5, 5), 1.0)
    t, _ = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    seed = cv2.morphologyEx(((sat > t) & (hue > 105) & (hue < 175)).astype(np.uint8), cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    h, w = seed.shape
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(seed)
    near = [k for k in range(1, n) if np.hypot(centroids[k, 0] - w / 2, centroids[k, 1] - h / 2) < 0.28 * w
            and stats[k, cv2.CC_STAT_AREA] > 40]
    nucleus = np.isin(labels, near).astype(np.uint8)
    if not nucleus.any():
        return nucleus
    ys, xs = np.nonzero(nucleus)
    yy, xx = np.mgrid[:h, :w]
    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    gc[np.hypot(yy - ys.mean(), xx - xs.mean()) < 1.8 * np.sqrt(nucleus.sum() / np.pi)] = cv2.GC_PR_FGD
    gc[nucleus > 0] = cv2.GC_FGD
    gc[:3, :] = gc[-3:, :] = gc[:, :3] = gc[:, -3:] = cv2.GC_BGD
    cv2.setRNGSeed(0)
    cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 5,
                cv2.GC_INIT_WITH_MASK)
    cell = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    rim = cv2.dilate(nucleus, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rim_px + 1, 2 * rim_px + 1)))
    cell[((hue < 30) | (hue > 172)) & (hsv[..., 1] > 40) & (rim == 0)] = 0
    cell = cv2.morphologyEx(cell, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    _, labels = cv2.connectedComponents(cell)
    ids = np.unique(labels[(nucleus > 0) & (cell > 0)])
    cell = np.isin(labels, ids[ids > 0]).astype(np.uint8)
    outside = np.pad(cell, 1)
    cv2.floodFill(outside, None, (0, 0), 1)
    return cell | (1 - outside[1:-1, 1:-1])
