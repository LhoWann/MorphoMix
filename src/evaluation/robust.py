"""Post-hoc robustness methods of the test-time scoring (`inference` in the config, `train.score_tests`), every one off
by default: WiSE-FT weight interpolation, Reinhard stain normalisation to the training cells, TENT entropy
minimisation (transductive) and the seed ensemble of prediction files."""
import copy
import json
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.augmentations.stain import MAX_SD_SCALE, MIN_REGION_PX, SD_FLOOR, from_unit, lab_to_rgb, region_moments, \
    rgb_to_lab, to_unit
from src.evaluation.field_inference import crop_batch, white_cell_mask
from src.evaluation.metrics import threshold_free_metrics
from src.models.factory import build_model

# WiSE-FT (Wortsman et al., CVPR 2022): theta = (1 - a) theta_pretrained + a theta_finetuned for every backbone
# tensor; the 2-class head has no pretrained counterpart, so the fine-tuned one is kept. `wise_alpha: val` takes the
# grid value with the highest val ROC-AUC (ties: the larger a, closer to the fine-tuned model)
WISE_ALPHAS = tuple(round(0.1 * i, 1) for i in range(1, 11))
HEAD_KEYS = ("head.fc.",)
# TENT (Wang et al., ICLR 2021): Adam (betas 0.9 / 0.999) on the LayerNorm affine parameters, entropy of the softmax,
# one view per input; batch fixed here so the adaptation does not depend on the hardware's `batch_size`
TENT_BATCH = 32
TENT_SEED = 0
_PRETRAINED: Dict[str, Dict[str, torch.Tensor]] = {}


def inference_record(inf: Dict) -> Dict:
    """`inference` without the keys of the robustness methods that are off, so settings recorded before these keys
    existed (`rescore` tags) still compare equal."""
    out = dict(inf)
    if out.get("wise_alpha") is None:
        out.pop("wise_alpha", None)
    if out.get("stain_norm", "none") == "none":
        out.pop("stain_norm", None)
    if not out.get("tent", False):
        for key in ("tent", "tent_steps", "tent_lr"):
            out.pop(key, None)
    return out


def summary(record: Dict) -> str:
    """Suffix of the `inference` label of a prediction file."""
    parts = []
    if "wise" in record:
        parts.append(f"WiSE-FT a {record['wise']['alpha']:g}")
    if "stain_norm" in record:
        parts.append(f"stain norm {record['stain_norm']}")
    if "tent" in record:
        parts.append("tent (transductive)")
    return "; " + ", ".join(parts) if parts else ""


# --- WiSE-FT ---

def pretrained_state(model_name: str) -> Dict[str, torch.Tensor]:
    """State dict of the timm weights every run starts from (`build_model(pretrained=True)`; its head is unused)."""
    if model_name not in _PRETRAINED:
        _PRETRAINED[model_name] = build_model(model_name, num_classes=2, pretrained=True).state_dict()
    return _PRETRAINED[model_name]


def wise_state(finetuned: Dict[str, torch.Tensor], pretrained: Dict[str, torch.Tensor], alpha: float
               ) -> Dict[str, torch.Tensor]:
    if alpha == 1.0:
        return finetuned
    return {k: v if k.startswith(HEAD_KEYS) or not v.is_floating_point()
            else torch.lerp(pretrained[k].to(v), v, alpha) for k, v in finetuned.items()}


def wise_ft(model: nn.Module, model_name: str, alpha, predict_val: Callable[[nn.Module], Dict]
            ) -> Tuple[nn.Module, Optional[Dict], Dict]:
    """A WiSE-FT copy of `model` at `alpha` (a number, or `val`: the WISE_ALPHAS value with the highest ROC-AUC of
    `predict_val`); returns the copy, the val predictions at the chosen alpha (None for a fixed alpha) and the
    record {alpha, val_auc per alpha}."""
    finetuned = {k: v.detach().clone() for k, v in model.state_dict().items()}
    pretrained = pretrained_state(model_name)
    out = copy.deepcopy(model)
    if alpha != "val":
        if isinstance(alpha, str):
            raise ValueError(f"inference.wise_alpha must be null, a number or 'val', got {alpha!r}")
        out.load_state_dict(wise_state(finetuned, pretrained, float(alpha)))
        return out, None, {"alpha": float(alpha)}
    preds, aucs = {}, {}
    for a in WISE_ALPHAS:
        out.load_state_dict(wise_state(finetuned, pretrained, a))
        preds[a] = predict_val(out)
        aucs[a] = threshold_free_metrics(preds[a]["y_true"], [r[1] for r in preds[a]["probs"]])["roc_auc"]
    best = max(WISE_ALPHAS, key=lambda a: (aucs[a], a))
    out.load_state_dict(wise_state(finetuned, pretrained, best))
    return out, preds[best], {"alpha": best, "rule": "val ROC-AUC", "val_auc": {str(a): aucs[a] for a in aucs}}


# --- stain normalisation ---

class ReinhardNormalizer:
    """Reinhard et al. (2001) in CIE Lab: the white-cell pixels of every image (`white_cell_mask`, the mask of field
    inference) move to the Lab mean and sd of the training cells (`mean_avg`, `sd_avg` of the RandStainNA fit of
    `prepare`, foreground of C-NMC), x -> (x - mean) * min(sd_train / sd, MAX_SD_SCALE) + mean_train per channel.
    Other pixels and images with fewer than MIN_REGION_PX cell pixels are left unchanged."""

    def __init__(self, stats_path: str, device: torch.device):
        with open(stats_path, encoding="utf-8") as f:
            stats = json.load(f)
        self.mean = torch.tensor(stats["mean_avg"], dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.sd = torch.tensor(stats["sd_avg"], dtype=torch.float32, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def rgb(self, rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """[B, 3, H, W] RGB in [0, 1] -> (normalised RGB, bool mask [B, 1, H, W] of the changed pixels)."""
        u8 = (rgb * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        mask = torch.from_numpy(np.stack([white_cell_mask(np.ascontiguousarray(im)) for im in u8]))
        mask = mask.to(rgb.device).bool().unsqueeze(1)
        lab = rgb_to_lab(rgb.float())
        mean, sd, area = region_moments(lab, (mask,))
        scale = (self.sd / sd[:, 0, :, None, None].clamp(min=SD_FLOOR)).clamp(max=MAX_SD_SCALE)
        out = lab_to_rgb((lab - mean[:, 0, :, None, None]) * scale + self.mean)
        mask = mask & (area[:, 0] >= MIN_REGION_PX).view(-1, 1, 1, 1)
        return out, mask

    def images(self, x: torch.Tensor) -> torch.Tensor:
        """ImageNet-normalised batch (the loaders' output)."""
        out, mask = self.rgb(to_unit(x))
        return torch.where(mask, from_unit(out).to(x.dtype), x)

    def field(self, u8: np.ndarray) -> np.ndarray:
        """uint8 RGB field [H, W, 3] (cell inference)."""
        rgb = torch.from_numpy(u8).to(self.mean.device).permute(2, 0, 1)[None].float() / 255
        out, mask = self.rgb(rgb)
        return (torch.where(mask, out, rgb)[0].permute(1, 2, 0) * 255).round().byte().cpu().numpy()


def stain_normalizer(inf: Dict, stats_path: str, device: torch.device) -> Optional[ReinhardNormalizer]:
    mode = inf.get("stain_norm", "none")
    if mode == "none":
        return None
    if mode != "reinhard":
        raise ValueError(f"inference.stain_norm must be none or reinhard, got {mode!r}")
    return ReinhardNormalizer(stats_path, device)


# --- TENT ---

def image_sampler(dataset, device: torch.device, transform: Optional[Callable] = None
                  ) -> Tuple[Callable[[Sequence[int]], torch.Tensor], int]:
    """(batch of dataset items by index, ImageNet-normalised on `device`, after `transform`; dataset size)."""
    def sample(idx: Sequence[int]) -> torch.Tensor:
        x = torch.stack([dataset[i][0] for i in idx]).to(device)
        return transform(x) if transform is not None else x
    return sample, len(dataset)


def crop_sampler(fields: List[np.ndarray], cells: List[np.ndarray], scale: float, out_size: int,
                 device: torch.device) -> Tuple[Callable[[Sequence[int]], torch.Tensor], int]:
    """(batch of cell crops by index over every detected cell of every field, as `score_crops` sees them; count)."""
    jobs = [(i, c) for i, cs in enumerate(cells) for c in cs]
    return (lambda idx: crop_batch(fields, [jobs[i] for i in idx], scale, out_size, device)), len(jobs)


def tent_adapt(model: nn.Module, sample: Callable[[Sequence[int]], torch.Tensor], n: int, steps: int, lr: float,
               autocast: Optional[Callable] = None, channels_last: bool = False) -> nn.Module:
    """An adapted copy of `model` (episodic: the caller's model is never changed). `steps` Adam steps of batch
    TENT_BATCH on the mean softmax entropy of unlabelled inputs, drawn in a fixed permutation (TENT_SEED) of the n
    inputs, cycled as needed; only LayerNorm weights and biases train, in eval mode (no stochastic depth)."""
    adapted = copy.deepcopy(model).eval()
    adapted.requires_grad_(False)
    params = [p for m in adapted.modules() if isinstance(m, nn.LayerNorm) for p in (m.weight, m.bias) if p is not None]
    for p in params:
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
    generator = torch.Generator().manual_seed(TENT_SEED)
    order = torch.cat([torch.randperm(n, generator=generator) for _ in range(math.ceil(steps * TENT_BATCH / n))])
    for s in range(steps):
        x = sample(order[s * TENT_BATCH:(s + 1) * TENT_BATCH].tolist())
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with autocast() if autocast is not None else torch.autocast(x.device.type, enabled=False):
            logits = adapted(x).float()
        loss = -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    adapted.requires_grad_(False)
    return adapted.eval()


# --- seed ensemble ---

def average_predictions(preds: List[Dict]) -> Dict:
    """Mean p(ALL) over the prediction files of one cohort from several checkpoints (same images, same order)."""
    names = preds[0]["names"]
    if any(p["names"] != names for p in preds[1:]):
        raise ValueError("ensemble members list other images or another order")
    p = np.mean([np.asarray(q["probs"], dtype=np.float64)[:, 1] for q in preds], axis=0)
    return {"y_true": preds[0]["y_true"], "names": names, "probs": np.stack([1 - p, p], 1).tolist()}
