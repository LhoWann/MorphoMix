"""Frozen foundation-model baselines (`python main.py foundation`): DinoBloom (Koch et al., MICCAI 2024), a ViT
pretrained on 13 haematology datasets (none of our training, selection or test cohorts; one internal set appears to
hold the MLL23 images). Features of the training cells (C-NMC plus the auxiliary cells, eval transforms, no
augmentation) fit a class-weighted logistic-regression probe whose C is chosen on val ROC-AUC; the probe on the
frozen backbone is then scored with the inference every arm gets (`train.score_tests`: dihedral TTA, Aria cell by
cell). Predictions: results_dir/predictions/phase1_{name}_seed{primary_seed}_*.json."""
import argparse
import json
import os
from typing import Dict, List, Optional

from src.utils.config import abs_path, load_config, refuse_real_results_dir, setup_cuda_env
setup_cuda_env()

import numpy as np  # noqa: E402
import timm  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from torch import nn  # noqa: E402

from scripts import train  # noqa: E402
from src.augmentations.transforms import IMAGENET_MEAN, IMAGENET_STD, get_val_transforms  # noqa: E402
from src.datasets.dataset import BinaryLeukemiaDataset  # noqa: E402
from src.datasets.loaders import eval_loader  # noqa: E402
from src.evaluation.predict import autocast_for  # noqa: E402
from src.utils.logger import get_console  # noqa: E402
from src.utils.seed import setup_run  # noqa: E402

MODELS = {
    "dinobloom_s": "hf-hub:1aurent/vit_small_patch14_224.dinobloom",
    "dinobloom_b": "hf-hub:1aurent/vit_base_patch14_224.dinobloom",
}
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


class ProbeModel(nn.Module):
    """Frozen backbone, the probe's standardisation and logistic weights; returns [B, 2] logits on inputs
    normalised with the ImageNet statistics our loaders use, re-normalised to the backbone's own."""

    def __init__(self, backbone: nn.Module, scaler: StandardScaler, probe: LogisticRegression):
        super().__init__()
        self.backbone = backbone
        cfg = backbone.pretrained_cfg
        to_t = lambda v: torch.tensor(v, dtype=torch.float32).view(1, -1, 1, 1)  # noqa: E731
        self.register_buffer("in_mean", to_t(IMAGENET_MEAN))
        self.register_buffer("in_std", to_t(IMAGENET_STD))
        self.register_buffer("own_mean", to_t(cfg.get("mean", IMAGENET_MEAN)))
        self.register_buffer("own_std", to_t(cfg.get("std", IMAGENET_STD)))
        self.register_buffer("f_mean", torch.tensor(scaler.mean_, dtype=torch.float32))
        self.register_buffer("f_scale", torch.tensor(scaler.scale_, dtype=torch.float32))
        self.register_buffer("coef", torch.tensor(probe.coef_[0], dtype=torch.float32))
        self.register_buffer("intercept", torch.tensor(float(probe.intercept_[0]), dtype=torch.float32))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(((x * self.in_std + self.in_mean) - self.own_mean) / self.own_std).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = (self.features(x) - self.f_mean) / self.f_scale
        logit = (z * self.coef).sum(1) + self.intercept  # fp32: a matmul would run in bf16 under autocast
        return torch.stack([torch.zeros_like(logit), logit], dim=1)


@torch.no_grad()
def extract(model: ProbeModel, loader, device: torch.device, autocast) -> tuple:
    feats, labels = [], []
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        with autocast() if autocast else torch.no_grad():
            feats.append(model.features(x).cpu().numpy())
        labels.append(batch[1].numpy())
    return np.concatenate(feats), np.concatenate(labels)


def run_foundation(cfg: Dict, name: str, limit: int, console) -> Dict:
    seed = int(cfg["primary_seed"])
    setup_run(cfg, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast = autocast_for(cfg["mixed_precision"], device)
    backbone = timm.create_model(MODELS[name], pretrained=True, num_classes=0).eval()
    dummy = ProbeModel(backbone, StandardScaler().fit(np.zeros((2, backbone.num_features))),
                       LogisticRegression().fit(np.zeros((2, backbone.num_features)), [0, 1])).to(device).eval()

    root = abs_path(cfg["phase1"]["data_dir"])
    extra = [abs_path(cfg["aux_train"]["dir"])] if cfg["aux_train"]["enabled"] else []
    fit_ds = BinaryLeukemiaDataset(os.path.join(root, train.TRAIN_DIR), get_val_transforms(cfg["img_size"]),
                                   limit=limit, extra_dirs=extra)
    datasets, loaders = train.build_loaders(cfg, seed, limit, aug="basic")
    x_tr, y_tr = extract(dummy, eval_loader(fit_ds, cfg), device, autocast)
    x_va, y_va = extract(dummy, loaders["val"], device, autocast)

    scaler = StandardScaler().fit(x_tr)
    grid = {}
    for c in C_GRID:
        probe = LogisticRegression(C=c, class_weight="balanced", max_iter=5000).fit(scaler.transform(x_tr), y_tr)
        grid[c] = float(roc_auc_score(y_va, probe.predict_proba(scaler.transform(x_va))[:, 1]))
        console.print(f"  {name} C={c:g}: val ROC-AUC {grid[c]:.4f}")
    best_c = max(grid, key=grid.get)
    probe = LogisticRegression(C=best_c, class_weight="balanced", max_iter=5000).fit(scaler.transform(x_tr), y_tr)
    model = ProbeModel(backbone, scaler, probe).to(device).eval()

    eid = f"phase1_{name}_seed{seed}"
    scores, summary = train.score_tests(cfg, eid, model, device, datasets, loaders, autocast=autocast)
    console.print(f"[bold]{eid}[/bold] C={best_c:g} | {' | '.join(summary)}")
    record = {"model": MODELS[name], "features": int(backbone.num_features), "n_train": int(len(y_tr)),
              "c_grid_val_roc_auc": {str(k): v for k, v in grid.items()}, "c": best_c, **scores}
    out = abs_path(os.path.join(cfg["results_dir"], "tables", f"foundation_{name}.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return record


def foundation_main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["dinobloom_s", "dinobloom_b"], choices=sorted(MODELS))
    p.add_argument("--limit", type=int, default=0, help="truncate the splits (smoke test)")
    a = p.parse_args(argv)
    cfg = load_config()
    if a.limit:
        refuse_real_results_dir(cfg["results_dir"])
    console = get_console()
    for name in a.models:
        run_foundation(cfg, name, a.limit, console)
    return 0
