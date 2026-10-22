"""Azure-B blast-cell colour prior."""
from typing import Tuple

import cv2
import numpy as np

# every setting the prior depends on; the MLL23 style bank records them (style_bank.bank_provenance)
PRIOR = {"hue": (125, 165), "min_azure_b": 15, "min_red_over_green": 10, "min_sat": 30, "margin": 6,
         "close_px": 7, "dilate_px": 5, "min_area_px": 120, "blur_px": 11, "blur_sigma": 3.0}


def extract_blast_cell_prior(rgb_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Soft cell prior, binary cell mask and chromatin map of an RGB uint8 image."""
    img_f = rgb_image.astype(np.float32)
    r, g, b = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]

    # Azure B contrast: (R + B)/2 - G
    azure_b = (r + b) / 2.0 - g

    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]

    # Giemsa purple blast
    is_purple = ((h >= PRIOR["hue"][0]) & (h <= PRIOR["hue"][1]) & (azure_b > PRIOR["min_azure_b"])
                 & (r > g + PRIOR["min_red_over_green"]) & (s > PRIOR["min_sat"]))
    mask = is_purple.astype(np.uint8) * 255

    # Clear sensor edge margin (lens chromatic aberration)
    margin = PRIOR["margin"]
    if margin > 0:
        mask[:margin, :] = 0
        mask[-margin:, :] = 0
        mask[:, :margin] = 0
        mask[:, -margin:] = 0

    # Close chromatin inside each cell
    kernel_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (PRIOR["close_px"],) * 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_c)
    kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (PRIOR["dilate_px"],) * 2)
    mask = cv2.dilate(mask, kernel_d, iterations=1)

    # Drop platelets / dust
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= PRIOR["min_area_px"]
    keep[0] = False  # background label
    clean_mask = np.where(keep[labels], 255, 0).astype(mask.dtype)

    gt_binary = (clean_mask > 0).astype(np.uint8)
    blur = (PRIOR["blur_px"],) * 2
    soft_prior = cv2.GaussianBlur(clean_mask.astype(np.float32) / 255.0, blur, PRIOR["blur_sigma"])

    chromatin = np.clip(azure_b, 0.0, 255.0) / 255.0
    chromatin = chromatin * gt_binary.astype(np.float32)
    if chromatin.max() > 1e-6:
        chromatin = chromatin / chromatin.max()

    return soft_prior.astype(np.float32), gt_binary, chromatin.astype(np.float32)


def prior_tensor_from_rgb(rgb_image: np.ndarray) -> np.ndarray:
    """[1, H, W] float32 soft cell prior - what the loader workers hand to PriorOnlyCAM."""
    return extract_blast_cell_prior(rgb_image)[0][None]
