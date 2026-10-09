"""The MLL23 style bank: per-image Lab moments of the nucleus and the cytoplasm (MorphoMix C1) and stain matrices
(the Stain Mix-up baseline) of every MLL23 cell, built once by `python main.py prepare` from data/raw/MLL23.

MLL23 labels are never read and no MLL23 image reaches the classifier: the bank holds statistics only.
"""
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.augmentations.background import CELL_LUMINANCE_FLOOR
from src.augmentations.stain import MIN_REGION_PX, REGIONS, region_moments, rgb_to_lab, split_nucleus
from src.augmentations.stain_baselines import MACENKO_ALPHA, MACENKO_BETA, MIN_STAIN_PX, estimate_stain_matrix
from src.cam.blast_prior import PRIOR
from src.datasets.cells import CENTRE_OPEN_PX, NUCLEUS_OPEN_PX, cell_mask, list_images, load_rgb, nucleus_cell
from src.utils.config import abs_path

BANK_VERSION = 3
# a row whose region log sd lies this many robust sds (median, 1.4826 MAD) above the bank's is a region failure: on
# inspection a cytoplasm that is only the nucleus rim, a pyknotic normoblast nucleus split in two, or two merged
# cells. Its sd, not its mean, makes Reinhard stretch the target's contrast into coloured noise. Region means are not
# filtered: their extremes are real stain and illumination styles (dim, grey captures), which C1 exists to transfer.
SD_ROBUST_Z_MAX = 4.0
BANK_KEYS = ("lab_mean", "lab_sd", "stain_matrix")  # the arrays the content digest covers, in this order
CHUNK = 256  # images per torch batch while building


def provenance_path(path: str) -> str:
    return os.path.splitext(path)[0] + ".json"


def bank_provenance(cfg: Dict) -> Dict:
    """What the bank is built from; `prepare` rebuilds it whenever this differs from the saved sidecar. C1's matching
    constants (MAX_SD_SCALE, SEAM_PX) act on the training cells only, so the bank does not depend on them."""
    bank = cfg["style_bank"]
    root = abs_path(bank["source_dir"])
    paths = list_images(root)
    names = "\n".join(os.path.relpath(p, root).replace(os.sep, "/") for p in paths)
    return {"version": BANK_VERSION, "source": bank["source_dir"], "n_images": len(paths),
            "files_sha256": hashlib.sha256(names.encode()).hexdigest(), "img_size": cfg["img_size"],
            "morpho_threshold": cfg["morpho_threshold"], "prior": PRIOR, "cell_luminance_floor": CELL_LUMINANCE_FLOOR,
            "mask": {"mode": "nucleus", "ring": bank["ring"], "min_sat": bank["min_sat"],
                     "nucleus_open_px": NUCLEUS_OPEN_PX, "centre_open_px": CENTRE_OPEN_PX},
            "macenko": {"beta": MACENKO_BETA, "alpha": MACENKO_ALPHA, "min_stain_px": MIN_STAIN_PX},
            "min_region_px": MIN_REGION_PX, "sd_robust_z_max": SD_ROBUST_Z_MAX}


def content_digest(arrays: Dict[str, np.ndarray]) -> str:
    """SHA-256 of the bank arrays (an npz file's own bytes change with its zip timestamps)."""
    h = hashlib.sha256()
    for key in BANK_KEYS:
        a = np.ascontiguousarray(arrays[key])
        h.update(f"{key}{a.dtype.str}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def bank_digest(path: str) -> str:
    """Content SHA-256 of a bank file, recorded with every run that drew from it."""
    with np.load(abs_path(path)) as z:
        return content_digest({k: z[k] for k in BANK_KEYS})


def _cell(path: str, cfg: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """uint8 RGB, the MLL23 cell mask (centre nucleus + cytoplasm ring) and the cell's stain matrix (3, 2)."""
    bank = cfg["style_bank"]
    rgb = load_rgb(path, cfg["img_size"])
    cell = nucleus_cell(rgb, cell_mask(rgb, cfg["morpho_threshold"]), bank["ring"], bank["min_sat"]) > 0
    return rgb, cell, np.asarray(estimate_stain_matrix(rgb, cell), dtype=np.float32)


@torch.no_grad()
def build_style_bank(cfg: Dict, threads: int = 3) -> Dict:
    """Writes the bank (npz) and its provenance sidecar (json); returns the provenance. Deterministic: images in
    sorted path order; a row only when both regions hold MIN_REGION_PX pixels, the stain matrix is finite and no
    region log sd exceeds SD_ROBUST_Z_MAX robust sds."""
    out = abs_path(cfg["style_bank"]["path"])
    root = abs_path(cfg["style_bank"]["source_dir"])
    paths = list_images(root)
    if not paths:
        raise FileNotFoundError(f"no MLL23 images under {root}")
    rows: Dict[str, List[np.ndarray]] = {k: [] for k in BANK_KEYS}
    kept: List[str] = []
    with ThreadPoolExecutor(threads) as pool:
        for start in range(0, len(paths), CHUNK):
            chunk = paths[start:start + CHUNK]
            rgb, cell, stain = (np.stack(x) for x in zip(*pool.map(lambda p: _cell(p, cfg), chunk)))
            x = torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255.0
            inside = torch.from_numpy(cell).unsqueeze(1)
            nucleus = split_nucleus(x, inside.float())
            mean, sd, area = region_moments(rgb_to_lab(x), (nucleus, inside & ~nucleus))
            ok = (area >= MIN_REGION_PX).all(dim=1).numpy() & np.isfinite(stain).all(axis=(1, 2))
            rows["lab_mean"].append(mean.numpy()[ok])
            rows["lab_sd"].append(sd.numpy()[ok])
            rows["stain_matrix"].append(stain[ok])
            kept += [os.path.relpath(p, root).replace(os.sep, "/") for p, k in zip(chunk, ok) if k]
            print(f"\rstyle bank {start + len(chunk):,} / {len(paths):,}", end="", flush=True)
    print()
    arrays = {k: np.concatenate(v).astype(np.float32) for k, v in rows.items()}
    log_sd = np.log(arrays["lab_sd"].reshape(len(kept), -1).astype(np.float64))
    median = np.median(log_sd, axis=0)
    z = (log_sd - median) / (1.4826 * np.median(np.abs(log_sd - median), axis=0))
    keep = (z <= SD_ROBUST_Z_MAX).all(axis=1)
    arrays = {k: v[keep] for k, v in arrays.items()}
    kept = [p for p, k in zip(kept, keep) if k]
    provenance = {**bank_provenance(cfg), "n_valid": len(kept), "n_dropped_region_sd": int((~keep).sum()),
                  "regions": list(REGIONS), "content_sha256": content_digest(arrays)}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".tmp", "wb") as f:
        np.savez_compressed(f, **arrays, paths=np.array(kept))
    os.replace(out + ".tmp", out)
    with open(provenance_path(out), "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    return provenance


class StyleBank(nn.Module):
    """C1's references: real MLL23 cells (Lab mean and sd per region, [N, 2, 3]) and virtual templates.

    A virtual template (RandStainNA-style, Shen et al., MICCAI 2022) is drawn from a Gaussian fitted to the bank in
    (mean, log sd) space, 12 dimensions, with the Ledoit-Wolf shrunk full covariance: the nucleus and cytoplasm
    moments of one slide move together (one stain bath, one camera), so independent per-statistic draws would pair a
    pale nucleus with a dark cytoplasm; the log keeps every sd positive. With 41,146 rows the shrinkage is negligible;
    it only matters for a small bank.
    """

    def __init__(self, lab_mean: np.ndarray, lab_sd: np.ndarray, digest: str):
        super().__init__()
        from sklearn.covariance import LedoitWolf
        self.digest = digest
        self.register_buffer("mean", torch.from_numpy(np.ascontiguousarray(lab_mean, dtype=np.float32)))
        self.register_buffer("sd", torch.from_numpy(np.ascontiguousarray(lab_sd, dtype=np.float32)))
        n = len(lab_mean)
        features = np.concatenate([lab_mean.reshape(n, -1), np.log(np.maximum(lab_sd.reshape(n, -1), 1e-3))], axis=1)
        fit = LedoitWolf().fit(features.astype(np.float64))
        self.register_buffer("template_mean", torch.from_numpy(fit.location_).float())
        self.register_buffer("template_chol", torch.from_numpy(np.linalg.cholesky(fit.covariance_)).float())

    @classmethod
    def load(cls, path: str) -> "StyleBank":
        with np.load(abs_path(path)) as z:
            arrays = {k: z[k] for k in BANK_KEYS}
        return cls(arrays["lab_mean"], arrays["lab_sd"], content_digest(arrays))

    def __len__(self) -> int:
        return len(self.mean)

    def virtual(self, b: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """b virtual templates: Lab mean and sd [b, 2, 3]."""
        z = self.template_mean + torch.randn(b, self.template_mean.numel(), device=device) @ self.template_chol.T
        return z[:, :6].view(b, 2, 3), z[:, 6:].exp().view(b, 2, 3)

    def draw(self, b: int, virtual_prob: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reference Lab mean and sd [b, 2, 3]: a random bank row, or with `virtual_prob` a virtual template."""
        row = torch.randint(len(self.mean), (b,), device=device)
        v_mean, v_sd = self.virtual(b, device)
        virtual = (torch.rand(b, device=device) < virtual_prob).view(b, 1, 1)
        return torch.where(virtual, v_mean, self.mean[row]), torch.where(virtual, v_sd, self.sd[row])
