"""CAM rendering helpers: heatmap overlays, mask contours, panel grids."""
import os
from typing import List, Sequence, Tuple

import numpy as np


def overlay_heatmap(rgb_img: np.ndarray, cam_map: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """JET heatmap blended with opacity proportional to the CAM, so zero-CAM pixels show the plain image."""
    import cv2
    cam_norm = np.clip(cam_map, 0.0, 1.0)
    heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam_norm), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    a = (alpha * cam_norm)[..., None]
    overlay = np.float32(heatmap) * a + np.float32(rgb_img) * (1.0 - a)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def draw_contour(rgb_img: np.ndarray, binary_mask: np.ndarray, color=(34, 139, 34)) -> np.ndarray:
    import cv2
    out = rgb_img.copy()
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


def render_panel_grid(
    grid: List[List[np.ndarray]],
    col_titles: Sequence[str],
    row_labels: Sequence[str],
    save_path: str,
    cell_captions: List[List[str]] = None,
    cell_size: Tuple[float, float] = (2.8, 2.8),
    suptitle: str = ""
) -> str:
    """Render a grid of RGB uint8 panels, grid[r][c]."""
    import matplotlib.pyplot as plt
    from src.utils import figstyle
    figstyle.apply()

    n_rows, n_cols = len(grid), len(col_titles)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(cell_size[0] * n_cols, cell_size[1] * n_rows))
    axes = np.atleast_2d(axes)
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title)
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.imshow(grid[r][c])
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(row_labels[r])
            if cell_captions and cell_captions[r][c]:
                ax.set_xlabel(cell_captions[r][c], fontsize=7)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    return save_path
