"""Test-time prediction of a trained classifier: whole images with dihedral TTA, or whole smear fields scored cell
by cell at the training scale (`field_inference`)."""
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.datasets.dataset import BinaryLeukemiaDataset
from src.evaluation.field_inference import crop_scales, detect_cells, field_scores, score_crops
from src.evaluation.tta import tta_logits


@torch.no_grad()
def predict_images(model: torch.nn.Module, loader: DataLoader, tta_views: int, device: torch.device,
                   autocast: Optional[Callable] = None, channels_last: bool = False,
                   transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> Dict:
    """{y_true, probs [N, 2], names} of every image in `loader`, the mean softmax over `tta_views` views;
    `transform` maps each normalised batch on `device` first (test-time stain normalisation)."""
    model.eval()
    y_true, probs, names = [], [], []
    for batch in loader:
        images = batch[0].to(device, non_blocking=True)
        if transform is not None:
            images = transform(images)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        with autocast() if autocast is not None else torch.autocast(device.type, enabled=False):
            p = tta_logits(model, images, n_views=tta_views).float().exp()
        probs.extend(p.cpu().numpy().tolist())
        y_true.extend(batch[1].tolist())
        names.extend(batch[2])
    return {"y_true": y_true, "probs": probs, "names": names}


def field_inputs(dataset: BinaryLeukemiaDataset, settings: Dict, transform: Optional[Callable] = None
                 ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """The uint8 RGB fields of `dataset`, mapped by `transform` when given (test-time stain normalisation), and
    their white cells, detected on the original fields."""
    fields = [np.asarray(Image.open(path).convert("RGB")) for path, _ in dataset.samples]
    whole = {k: float(settings.get(k, 0.0)) for k in ("min_inside", "min_nucleus")}
    cells = [detect_cells(f, float(settings["min_radius"]), **whole) for f in fields]
    return ([transform(f) for f in fields] if transform is not None else fields), cells


def predict_fields(model: torch.nn.Module, dataset: BinaryLeukemiaDataset, whole: Dict, settings: Dict,
                   tta_views: int, device: torch.device, img_size: int, autocast: Optional[Callable] = None,
                   channels_last: bool = False, batch_size: int = 128,
                   inputs: Optional[Tuple[List[np.ndarray], List[np.ndarray]]] = None) -> Dict:
    """`whole` (the `predict_images` output of the same cohort) with the field scores of `cells` inference: the
    white cells of every field are detected, cropped at `settings['scale']` (a list: crop p(ALL) averaged over the
    scales) and scored, and their p(ALL) aggregated (`settings['aggregate']`); a field without a detected cell keeps
    its whole-field score. `inputs` is a precomputed `field_inputs`. The per-field crop scores are returned as
    `cell_probs`."""
    fields, cells = inputs if inputs is not None else field_inputs(dataset, settings)
    per_scale = [score_crops(model, fields, cells, scale, img_size, tta_views, device, batch_size=batch_size,
                             autocast=autocast, channels_last=channels_last) for scale in crop_scales(settings)]
    cell_probs = per_scale[0] if len(per_scale) == 1 else [np.mean(p, axis=0) for p in zip(*per_scale)]
    p = field_scores(cell_probs, np.asarray(whole["probs"])[:, 1], settings["aggregate"], int(settings["top_k"]))
    return {**whole, "probs": np.stack([1 - p, p], 1).tolist(), "cell_probs": [c.tolist() for c in cell_probs]}


def autocast_for(mixed_precision: str, device: torch.device) -> Optional[Callable]:
    """Autocast factory for config `mixed_precision` (bf16 | fp16 | no) outside a Trainer, None when off."""
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(str(mixed_precision))
    return (lambda: torch.autocast(device.type, dtype=dtype)) if dtype is not None else None
