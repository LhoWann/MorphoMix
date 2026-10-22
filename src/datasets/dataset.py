"""Binary Normal / ALL image-folder dataset and the Aria annotation mask."""
import os
import glob
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


ANNOTATION_FILL = (220, 220, 220)  # one constant for every image, so the patch carries no class information
EXTENSIONS = ("*.jpg", "*.png", "*.jpeg", "*.bmp", "*.tif", "*.tiff")


def mask_annotation(image: Image.Image, box: Sequence[float]) -> Image.Image:
    """Paint the scale-bar corner, box = (y0, y1, x0, x1) as fractions of the image, in ANNOTATION_FILL.

    Aria et al. annotate every image in that corner and the annotation differs by class, so it is removed
    before the backbone or the XAI pseudo ground truth ever sees the image. A per-image fill (e.g. the median)
    would re-encode the image's global colour, which on its own predicts the class.
    """
    a = np.array(image)
    h, w = a.shape[:2]
    y0, y1 = int(box[0] * h), int(round(box[1] * h))
    x0, x1 = int(box[2] * w), int(round(box[3] * w))
    a[y0:y1, x0:x1] = np.asarray(ANNOTATION_FILL, dtype=a.dtype)
    return Image.fromarray(a)


def _prior_for(image: torch.Tensor) -> torch.Tensor:
    """Azure-B prior of an already-transformed (normalized) image."""
    from src.augmentations.transforms import denormalize_image
    from src.cam.blast_prior import prior_tensor_from_rgb
    return torch.from_numpy(prior_tensor_from_rgb(denormalize_image(image)))


def stratified_subset(samples: List[Tuple[str, int]], fraction: float, seed: int) -> List[Tuple[str, int]]:
    """A random `fraction` of every (label, fold) stratum, the fold read from a `fold_<k>` file-name part (C-NMC); the
    subset depends only on `seed` and the file list, so it is the same for every candidate and run seed."""
    rng = np.random.default_rng(seed)
    strata: Dict[Tuple[int, str], List[int]] = {}
    for i, (path, label) in enumerate(samples):
        fold = re.search(r"fold_(\d+)", os.path.basename(path))
        strata.setdefault((label, fold[1] if fold else ""), []).append(i)
    keep = []
    for key in sorted(strata):
        members = strata[key]
        keep += [members[j] for j in rng.permutation(len(members))[:max(1, round(fraction * len(members)))]]
    return [samples[i] for i in sorted(keep)]


class BinaryLeukemiaDataset(Dataset):
    """`base_dir/{Normal,ALL}` images, labels 0 / 1, followed by those of every `extra_dirs` entry (same layout);
    `fraction` < 1 keeps a stratified random subset of the union (`stratified_subset`)."""
    def __init__(
        self,
        base_dir: str,
        transform: Optional[Callable] = None,
        limit: int = 0,
        with_prior: bool = False,
        fraction: float = 1.0,
        subset_seed: int = 0,
        extra_dirs: Sequence[str] = (),
    ):
        self.base_dir = base_dir
        self.transform = transform
        self.with_prior = with_prior  # also return the Azure-B prior
        self.samples = []

        for root in (base_dir, *extra_dirs):
            for label, cls in enumerate(("Normal", "ALL")):
                cls_dir = os.path.join(root, cls)
                files = dict.fromkeys(os.path.normpath(f) for ext in EXTENSIONS
                                      for f in sorted(glob.glob(os.path.join(cls_dir, ext))))
                if not files:  # e.g. a partial unzip on Colab
                    raise FileNotFoundError(f"{cls_dir} is missing or holds no image; rebuild the cohort with "
                                            f"`python main.py prepare` or re-extract the data archive")
                self.samples += [(f, label) for f in files]

        if fraction < 1.0:
            self.samples = stratified_subset(self.samples, fraction, subset_seed)
        if limit > 0:
            # Class-balanced truncation for smoke tests
            per_cls = max(1, limit // 2)
            keep = [s for s in self.samples if s[1] == 0][:per_cls] + [s for s in self.samples if s[1] == 1][:per_cls]
            self.samples = keep

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.with_prior:
            return image, label, os.path.basename(path), _prior_for(image)
        return image, label, os.path.basename(path)
