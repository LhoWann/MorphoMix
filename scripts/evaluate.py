"""Evaluation built from artifacts on disk: result tables (`tables`, with the resource summary), re-scoring with
another test-time inference (`rescore`), paired bootstrap (`stats`), stress suite and val-only screening scores
(`stress`), backbone profile (`profile`) and XAI localisation on ALL-IDB2 (`xai`)."""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from src.utils.config import load_config, abs_path, apply_overrides, refuse_real_results_dir, setup_cuda_env
setup_cuda_env()

import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, matthews_corrcoef, roc_auc_score,
    roc_curve,
)
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.augmentations.transforms import (  # noqa: E402
    IMAGENET_MEAN, IMAGENET_STD, denormalize_image, get_val_transforms,
)
from src.cam.blast_prior import extract_blast_cell_prior  # noqa: E402
from src.cam.evidence_cam import EvidenceCAM  # noqa: E402
from src.cam.layer_cam import LayerCAM  # noqa: E402
from src.cam.metrics import central_cell_mask, evaluate_xai_metrics  # noqa: E402
from src.cam.visualizer import draw_contour, overlay_heatmap, render_panel_grid  # noqa: E402
from src.datasets.dataset import BinaryLeukemiaDataset  # noqa: E402
from src.datasets.loaders import eval_loader  # noqa: E402
from src.evaluation.calibration import decision_metrics, resolve_threshold  # noqa: E402
from src.evaluation.metrics import threshold_free_metrics  # noqa: E402
from src.evaluation.predict import autocast_for  # noqa: E402
from src.evaluation.robust import average_predictions, inference_record  # noqa: E402
from src.evaluation.stress import SELECTION, evaluate_families, load_cohort, selection_score  # noqa: E402
from src.models.factory import build_model, get_target_cam_layer  # noqa: E402
from src.models.profiler import benchmark_inference_latency, profile_model  # noqa: E402
from src.utils import figstyle  # noqa: E402
from src.utils.exporter import export_results_table  # noqa: E402
from src.utils.logger import get_console  # noqa: E402
from src.utils.resources import print_summary  # noqa: E402
from src.utils.seed import setup_run  # noqa: E402
from scripts import train  # noqa: E402
from scripts.train import TEST_SETS, VAL_DIR, seeds_for  # noqa: E402


def _ckpt_path(cfg: Dict, experiment_id: str) -> str:
    return os.path.join(abs_path(cfg["results_dir"]), "checkpoints", f"{experiment_id}_best.pt")


RESCORE_DIR = "rescore"  # under results_dir: {tag}/predictions, tables, figures of `rescore --tag TAG`


def _out_root(cfg: Dict, rescore: Optional[str] = None) -> str:
    """results_dir, or results_dir/rescore/{tag} for the predictions and tables of a re-scored set."""
    root = abs_path(cfg["results_dir"])
    return os.path.join(root, RESCORE_DIR, rescore) if rescore else root


def _pred_path(cfg: Dict, experiment_id: str, test_key: str, rescore: Optional[str] = None) -> str:
    return os.path.join(_out_root(cfg, rescore), "predictions", f"{experiment_id}_test_{test_key}.json")


def rule_threshold(cfg: Dict, rule, experiment_id: str, test_p: List[float], rescore: Optional[str] = None) -> float:
    """A decision rule (`inference.decision_threshold` / `secondary_threshold`) resolved at report time, so tables
    and stats follow the config rather than the threshold a prediction file was written with; `val_youden` reads
    the run's own val predictions ({id}_val.json)."""
    if rule == "val_youden":
        val = _load_json(os.path.join(_out_root(cfg, rescore), "predictions", f"{experiment_id}_val.json"))
        return resolve_threshold(rule, val["y_true"], [row[1] for row in val["probs"]])
    return resolve_threshold(rule, test_p=test_p)


def _tables(cfg: Dict, name: str, rescore: Optional[str] = None) -> str:
    return os.path.join(_out_root(cfg, rescore), "tables", name)


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_phase1_model(cfg: Dict, path: str, device) -> torch.nn.Module:
    """A trained checkpoint in eval mode on `device`."""
    model = build_model(cfg["model_name"], num_classes=len(cfg["phase1"]["class_names"]), pretrained=False,
                        dropout=cfg["dropout"])
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"])
    return model.to(device).eval()


def _val_summary(path: str) -> Optional[Dict]:
    """Validation metrics of the selected epoch, read back from the checkpoint."""
    if not os.path.exists(path):
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    metrics = state.get("metrics", {})
    return {
        "best_epoch": state.get("epoch"),
        "roc_auc": metrics.get("roc_auc"),
        "macro_f1": state.get("macro_f1", metrics.get("macro_f1")),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "val_loss": state.get("val_loss"),
    }


def phase1_result(cfg: Dict, experiment_id: str, rescore: Optional[str] = None) -> Optional[Dict]:
    """One run from its checkpoint (val) and predictions (test), keyed like `train.run_one`."""
    val = _val_summary(_ckpt_path(cfg, experiment_id))
    preds = {k: _pred_path(cfg, experiment_id, k, rescore) for k in TEST_SETS}
    if val is None or not all(os.path.exists(p) for p in preds.values()):
        return None
    r = dict(val)
    inf, names = cfg["inference"], cfg["phase1"]["class_names"]
    for key, path in preds.items():
        p = _load_json(path)
        test_p = [row[1] for row in p["probs"]]
        free = threshold_free_metrics(p["y_true"], test_p)
        # recomputed from the probabilities at the configured rule, not read from `y_pred` or the file's threshold
        dec = decision_metrics(p["y_true"], p["probs"], names,
                               threshold=rule_threshold(cfg, inf["decision_threshold"], experiment_id, test_p, rescore))
        if inf.get("secondary_threshold") is not None:
            sec = decision_metrics(p["y_true"], p["probs"], names, threshold=rule_threshold(
                cfg, inf["secondary_threshold"], experiment_id, test_p, rescore))
            r.update({f"test_{key}_threshold_secondary": sec["threshold"],
                      f"test_{key}_macro_f1_secondary": sec["macro_f1"],
                      f"test_{key}_predicted_positive_rate_secondary": sec["predicted_positive_rate"]})
        r.update({
            f"test_{key}_roc_auc": free["roc_auc"],
            f"test_{key}_auprc": free["auprc"],
            f"test_{key}_threshold": dec["threshold"],
            f"test_{key}_macro_f1": dec["macro_f1"],
            f"test_{key}_balanced_acc": dec["balanced_accuracy"],
            f"test_{key}_accuracy": dec["accuracy"],
            f"test_{key}_per_class_f1": dec["per_class_f1"],
            f"test_{key}_sensitivity_all": dec["per_class_recall"].get("ALL", 0.0),
            f"test_{key}_specificity_all": dec["per_class_specificity"].get("ALL", 0.0),
            f"test_{key}_predicted_positive_rate": dec["predicted_positive_rate"],
            f"test_{key}_degenerate": dec["degenerate"],
            f"test_{key}_inference": p.get("inference", "resize"),
        })
    return r


def rebuild_phase1(cfg: Dict, augs: List[str], seeds: List[int], console, rescore: Optional[str] = None
                   ) -> List[Dict]:
    results, missing = [], []
    for aug in augs:
        # Scheme B is a floor, not a cap
        required = seeds_for(cfg, aug, seeds)
        present = [s for s in seeds if os.path.exists(_ckpt_path(cfg, f"phase1_{aug}_seed{s}"))]
        for seed in sorted(set(required) | set(present)):
            r = phase1_result(cfg, f"phase1_{aug}_seed{seed}", rescore)
            if r is None:
                if seed in required:
                    missing.append(f"phase1_{aug}_seed{seed}")
                continue
            results.append({"augmentation": aug, "seed": seed, **r})
    if results:
        train._export_table(cfg, results, _out_root(cfg, rescore))
        console.print(f"ok result table rebuilt from {len(results)} runs -> {train.TABLE_PREFIX}.*")
    else:
        console.print("[warning]Result table: nothing to rebuild (no complete run found).[/warning]")
    if missing:
        console.print(f"[dim]Incomplete/absent runs: {', '.join(missing)}[/dim]")
    return results


def tables_main(argv=None):
    cfg = load_config()
    p = argparse.ArgumentParser(description="Rebuild the result tables from artifacts on disk.")
    p.add_argument("--aug", nargs="+", default=cfg["augmentations"])
    p.add_argument("--seed", nargs="+", type=int, default=cfg["seeds"])
    p.add_argument("--rescore", metavar="TAG", default=None, help="the predictions of `rescore --tag TAG`")
    a = p.parse_args(argv)
    console = get_console()
    rebuild_phase1(cfg, a.aug, a.seed, console, a.rescore)
    if a.rescore:
        return
    background_reference(cfg, console)
    print_summary(os.path.join(abs_path(cfg["results_dir"]), "logs"), console)


def rescore_main(argv=None) -> int:
    """Re-score finished checkpoints with the current test-time inference (config `inference`, `--set`) into
    results_dir/rescore/TAG, so the predictions of the training runs stay as they are; `tables --rescore TAG` and
    `stats --rescore TAG` read them. No training, no test label is used to choose anything."""
    cfg = load_config()
    p = argparse.ArgumentParser(description=rescore_main.__doc__)
    p.add_argument("--tag", required=True, help="name of the rescore set, results_dir/rescore/TAG")
    p.add_argument("--aug", nargs="+", default=cfg["augmentations"])
    p.add_argument("--seed", nargs="+", type=int, default=cfg["seeds"])
    p.add_argument("--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
                   help="config overrides, e.g. inference.field=cells inference.decision_threshold=val_youden")
    p.add_argument("--limit", type=int, default=0, help="truncate each cohort to N images (smoke test)")
    p.add_argument("--force", action="store_true", help="re-score runs whose predictions exist")
    p.add_argument("--ensemble", action="store_true",
                   help="also average p(ALL) over the seeds of each arm (val and tests), threshold rule on the "
                        "averaged val, into rescore/TAG/ensemble and tables/seed_ensemble")
    a = p.parse_args(argv)
    cfg = apply_overrides(cfg, {k: train.parse_value(v) for k, v in (x.split("=", 1) for x in a.set)})
    if a.limit:
        refuse_real_results_dir(cfg["results_dir"])
    console = get_console()
    root = _out_root(cfg, a.tag)
    views = cfg.get("tta_views", 8)
    settings = {"inference": inference_record(cfg["inference"]), "tta_views": views, "limit": a.limit}
    settings_path = os.path.join(root, "inference.json")
    if os.path.exists(settings_path) and _load_json(settings_path) != settings:
        raise SystemExit(f"{settings_path} holds other settings than {settings}; one tag holds one inference, use "
                         f"another --tag")
    os.makedirs(root, exist_ok=True)
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = abs_path(cfg["phase1"]["data_dir"])
    transform = get_val_transforms(cfg["img_size"])
    subs = {"val": VAL_DIR, **TEST_SETS}
    datasets = {k: BinaryLeukemiaDataset(os.path.join(data_root, d), transform, limit=a.limit) for k, d in subs.items()}
    loaders = {k: eval_loader(ds, cfg) for k, ds in datasets.items()}
    for aug in a.aug:
        for seed in a.seed:
            eid = f"phase1_{aug}_seed{seed}"
            path = _ckpt_path(cfg, eid)
            if not os.path.exists(path):
                console.print(f"[dim]{eid}: no checkpoint[/dim]")
                continue
            if not a.force and all(os.path.exists(_pred_path(cfg, eid, k, a.tag)) for k in TEST_SETS):
                console.print(f"[dim]{eid}: done, skipped[/dim]")
                continue
            setup_run(cfg, seed)  # deterministic kernels
            model = load_phase1_model(cfg, path, device)
            if cfg["channels_last"]:
                model = model.to(memory_format=torch.channels_last)
            _, summary = train.score_tests(cfg, eid, model, device, datasets, loaders,
                                           autocast=autocast_for(cfg["mixed_precision"], device), out_root=root)
            console.print(f"[bold]{eid}[/bold] {' | '.join(summary)}")
            del model
            torch.cuda.empty_cache()
    rebuild_phase1(cfg, a.aug, a.seed, console, a.tag)
    if a.ensemble:
        seed_ensemble(cfg, a.aug, a.seed, a.tag, console)
    return 0


def seed_ensemble(cfg: Dict, augs: List[str], seeds: List[int], rescore: str, console) -> List[Dict]:
    """Seed ensemble per arm from the predictions of `rescore --tag`: mean p(ALL) over the seeds with complete
    predictions (at least two), for val and every test cohort; the threshold rule (`decision_threshold`) reads the
    averaged val. Writes rescore/TAG/ensemble/predictions/phase1_{aug}_ensemble_{split}.json and
    tables/seed_ensemble."""
    root = _out_root(cfg, rescore)
    splits = ["val"] + [f"test_{k}" for k in TEST_SETS]

    def path(eid: str, split: str) -> str:
        return os.path.join(root, "predictions", f"{eid}_{split}.json")

    rows = []
    for aug in augs:
        ids = [f"phase1_{aug}_seed{s}" for s in seeds
               if all(os.path.exists(path(f"phase1_{aug}_seed{s}", split)) for split in splits)]
        if len(ids) < 2:
            continue
        eid = f"phase1_{aug}_ensemble"
        val = average_predictions([_load_json(path(i, "val")) for i in ids])
        val_p = [r[1] for r in val["probs"]]
        train._save_predictions(os.path.join(root, "ensemble"), eid, "val", val)
        val_auc = threshold_free_metrics(val["y_true"], val_p)["roc_auc"]
        row = {"augmentation": aug, "seeds": len(ids), "val_roc_auc": val_auc}
        for key in TEST_SETS:
            members = [_load_json(path(i, f"test_{key}")) for i in ids]
            test = average_predictions(members)
            test_p = [r[1] for r in test["probs"]]
            test["threshold"] = resolve_threshold(cfg["inference"]["decision_threshold"], val["y_true"], val_p, test_p)
            test["inference"] = f"{members[0].get('inference', 'resize')}; mean of {len(ids)} seeds"
            free = threshold_free_metrics(test["y_true"], test_p)
            dec = decision_metrics(test["y_true"], test["probs"], cfg["phase1"]["class_names"],
                                   threshold=test["threshold"])
            row.update({f"{key}_roc_auc": free["roc_auc"], f"{key}_auprc": free["auprc"],
                        f"{key}_macro_f1": dec["macro_f1"], f"{key}_threshold": dec["threshold"],
                        f"{key}_predicted_positive_rate": dec["predicted_positive_rate"]})
            train._save_predictions(os.path.join(root, "ensemble"), eid, f"test_{key}", test)
        rows.append({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
    if rows:
        export_results_table(rows, _tables(cfg, "seed_ensemble", rescore), label="tab:seed_ensemble", caption=(
            "Seed ensemble: mean p(ALL) over the seeds of each arm; ALL iff the mean >= the threshold of the "
            f"configured rule ({cfg['inference']['decision_threshold']}) on the averaged val predictions."))
        console.print(f"ok seed ensemble of {len(rows)} arms -> {_tables(cfg, 'seed_ensemble', rescore)}.*")
    return rows


def background_reference(cfg: Dict, console) -> List[Dict]:
    """How well the ALL-IDB2 background alone separates its classes: ROC-AUC (ALL positive) of simple colour
    statistics of the pixels outside the cell mask (the training mask rule). Nothing is fitted, so no direction is
    chosen on the test set; an AUC far from 0.5 either way means the background carries the label."""
    from sklearn.metrics import roc_auc_score
    from PIL import Image
    tau = float(cfg["morpho_threshold"])
    root = os.path.join(abs_path(cfg["phase1"]["data_dir"]), TEST_SETS["allidb2"])
    feats, labels = [], []
    for label, cls in enumerate(cfg["phase1"]["class_names"]):
        for f in sorted(glob.glob(os.path.join(root, cls, "*.png"))):
            rgb = np.asarray(Image.open(f).convert("RGB"))
            soft, _, _ = extract_blast_cell_prior(rgb)
            bg = (soft - soft.min()) / max(float(soft.max() - soft.min()), 1e-8) < tau
            px = rgb[bg].astype(np.float64) if bg.any() else rgb.reshape(-1, 3).astype(np.float64)
            r, g, b = px.mean(0)
            feats.append({"background R": r, "background G": g, "background B": b, "background B - R": b - r,
                          "background brightness": px.mean()})
            labels.append(label)
    if not feats:
        console.print("[warning]ALL-IDB2 background reference: no images found.[/warning]")
        return []
    rows = [{"feature": k, "roc_auc_all_positive": round(float(roc_auc_score(labels, [x[k] for x in feats])), 4)}
            for k in feats[0]]
    export_results_table(rows, _tables(cfg, "allidb2_background_reference"),
                         caption=("ALL-IDB2: ROC-AUC of background colour statistics alone (pixels outside the cell "
                                  "mask, nothing fitted; ALL = positive). An AUC far from 0.5 in either direction "
                                  "means a model can score on ALL-IDB2 from the background, without the cell."),
                         label="tab:allidb2_background")
    console.print(f"ok ALL-IDB2 background reference: max |AUC - 0.5| = "
                  f"{max(abs(r['roc_auc_all_positive'] - 0.5) for r in rows):.3f}")
    return rows


PRED_PATTERN = re.compile(r"^phase1_(?P<aug>[a-z_0-9]+?)_seed(?P<seed>\d+)_test_(?P<tset>[a-z0-9_]+)\.json$")


def prediction_files(cfg: Dict, rescore: Optional[str] = None) -> Dict[Tuple[str, str, int], str]:
    """(test set, aug, seed) -> path of every per-image test prediction file under results_dir (or the rescore
    set `rescore`)."""
    out = {}
    for path in glob.glob(os.path.join(_out_root(cfg, rescore), "predictions", "phase1_*_test_*.json")):
        m = PRED_PATTERN.match(os.path.basename(path))
        if m:
            out[(m.group("tset"), m.group("aug"), int(m.group("seed")))] = path
    return out


def stats_pairs(keys, reference: str, baselines: List[str]) -> Set[Tuple[str, int, str]]:
    """(test set, seed, baseline) of every comparison `stats` makes from prediction keys (test set, aug, seed)."""
    keys = set(keys)
    return {(t, s, b) for t, aug, s in keys if aug == reference for b in baselines if (t, b, s) in keys}


def load_predictions(cfg: Dict, rescore: Optional[str] = None) -> Dict[Tuple[str, str, int], Dict]:
    """Key = (test set, aug, seed) -> {y_true, y_pred, scores [N, 2], names}."""
    out = {}
    for key, path in prediction_files(cfg, rescore).items():
        d = _load_json(path)
        probs = np.asarray(d["probs"], dtype=np.float64)
        t = rule_threshold(cfg, cfg["inference"]["decision_threshold"], f"phase1_{key[1]}_seed{key[2]}",
                           probs[:, 1].tolist(), rescore)
        out[key] = {
            "y_true": np.asarray(d["y_true"], dtype=np.int64),
            # derived at the configured rule, not read from the file
            "y_pred": (probs[:, 1] >= t).astype(np.int64),
            "scores": probs,
            "names": d.get("names"),
        }
    return out


def load_groups(cfg: Dict, test_set: str) -> Optional[Dict[str, int]]:
    """{file name: group} of a test set from its CSV (`split`), None when an optional CSV is missing."""
    path = abs_path(cfg[GROUPED_TEST_SETS[test_set][0]]["csv"])
    if not os.path.exists(path):
        if test_set in REQUIRED_GROUPS:
            raise FileNotFoundError(f"{path} is missing; run `python main.py split`.")
        print(f"  {path} is missing: {test_set} is resampled by image")
        return None
    with open(path, encoding="utf-8", newline="") as f:
        return {row["file"]: int(row["group"]) for row in csv.DictReader(f)}


def group_strata(names: Optional[List[str]], y: np.ndarray, groups: Dict[str, int]) -> List[List[np.ndarray]]:
    """Image indices of each group, the groups split by class."""
    if not names:
        raise RuntimeError("Predictions carry no file names, so they cannot be mapped to groups.")
    missing = [n for n in names if n not in groups]
    if missing:
        raise KeyError(f"{len(missing)} images have no group (e.g. {missing[0]}); run `main.py split`.")
    by_group = defaultdict(list)
    for i, n in enumerate(names):
        by_group[groups[n]].append(i)
    strata = defaultdict(list)
    for g, idx in by_group.items():
        if len(set(y[idx].tolist())) > 1:
            raise ValueError(f"group {g} holds both classes; a resampling unit must not mix them")
        strata[int(y[idx[0]])].append(np.asarray(idx))
    return [strata[c] for c in sorted(strata)]


def macro_f1(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    """Macro-F1 from one confusion matrix."""
    cm = np.bincount(y * n_classes + pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(cm)
    denom = cm.sum(axis=0) + cm.sum(axis=1)  # 2tp + fp + fn
    present = denom > 0
    return float(np.mean(2 * tp[present] / denom[present]))


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney form, with ties averaged the way sklearn does."""
    pos = np.count_nonzero(y == 1)
    neg = y.size - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    starts = np.r_[True, s[1:] != s[:-1]]  # first position of each tie group
    first = np.flatnonzero(starts)
    last = np.r_[first[1:] - 1, s.size - 1]
    avg_rank = (first + last + 2) / 2.0  # 1-based mean rank within the group
    ranks = avg_rank[np.cumsum(starts) - 1]
    return float((ranks[y[order] == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """AUPRC."""
    pos = np.count_nonzero(y == 1)
    if pos == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    yy, ss = y[order], score[order]
    cut = np.r_[np.flatnonzero(np.diff(ss)), ss.size - 1]
    tp = np.cumsum(yy == 1)[cut]
    fp = (cut + 1) - tp
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


METRICS = {
    "macro_f1": lambda y, pred, s: macro_f1(y, pred, s.shape[1]),
    "roc_auc": lambda y, pred, s: roc_auc(y, s[:, 1]),
    "auprc": lambda y, pred, s: average_precision(y, s[:, 1]),
}
# images of one group are not independent test images, so the bootstrap resamples whole groups within class: Aria
# repeated captures of one smear field, ALL-IDB2 crops of one cell. test set: (config key of the CSV, unit name)
GROUPED_TEST_SETS = {"aria": ("aria_groups", "field group"), "allidb2": ("allidb2_groups", "cell group")}
REQUIRED_GROUPS = ("aria",)  # ALL-IDB2 falls back to single images without its CSV


def paired_bootstrap(ref: Dict, base: Dict, n_boot: int, rng, strata: Optional[List[List[np.ndarray]]] = None
                     ) -> Dict[str, Tuple[float, float, float, float]]:
    """Per metric (observed, ci_low, ci_high, p): the difference on the full test set, its bootstrap CI and p-value.

    One set of resampled indices serves every metric and both arms. With `strata` (clusters of image indices, per
    class), whole clusters are resampled within each class instead of single images. `p` is the two-sided bootstrap
    p-value of a zero difference, floored at 1 / (B + 1) for B valid resamples.
    """
    y = ref["y_true"]
    n = y.size
    diffs = {k: np.empty(n_boot) for k in METRICS}
    for i in range(n_boot):
        if strata is None:
            idx = rng.integers(0, n, n)
        else:
            idx = np.concatenate([s[j] for s in strata for j in rng.integers(0, len(s), len(s))])
        y_i = y[idx]
        for name, fn in METRICS.items():
            diffs[name][i] = (fn(y_i, ref["y_pred"][idx], ref["scores"][idx])
                              - fn(y_i, base["y_pred"][idx], base["scores"][idx]))
    out = {}
    for name, d in diffs.items():
        d = d[~np.isnan(d)]  # a resample can lose a class
        fn = METRICS[name]
        observed = fn(y, ref["y_pred"], ref["scores"]) - fn(y, base["y_pred"], base["scores"])
        p = min(1.0, max(1.0 / (d.size + 1), 2.0 * min(float(np.mean(d <= 0)), float(np.mean(d >= 0)))))
        out[name] = (float(observed), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), p)
    return out


def holm(p_values: List[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values (step-down, monotone), in the input order."""
    m = len(p_values)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda k: p_values[k])):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def _apply_holm(rows: List[Dict]) -> None:
    """Adjust within each (test set, seed) family: every baseline x metric compared on one test set."""
    families = defaultdict(list)
    for r in rows:
        for name in METRICS:
            families[(r["test_set"], r["seed"])].append((r, name))
    for members in families.values():
        adjusted = holm([r[f"{name}_p"] for r, name in members])
        for (r, name), p_adj in zip(members, adjusted):
            r[f"{name}_p_holm"] = round(p_adj, 4)
            r[f"{name}_significant"] = "yes" if p_adj < 0.05 else "no"


def stats_main(argv=None):
    cfg = load_config()
    p = argparse.ArgumentParser(description="Paired bootstrap of MorphoMix against every baseline arm.")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--reference", default="morpho_mix")
    p.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    p.add_argument("--rescore", metavar="TAG", default=None, help="the predictions of `rescore --tag TAG`")
    a = p.parse_args(argv)

    preds = load_predictions(cfg, a.rescore)
    if not preds:
        raise SystemExit("No predictions found; run `main.py pretrain` first.")
    # never the ablation arms: they would widen every Holm family
    baselines = [x for x in cfg["augmentations"] if x != a.reference]
    for tset, aug, seed in sorted(preds):
        absent = [b for b in baselines if aug == a.reference and (tset, b, seed) not in preds]
        if absent:
            print(f"  warning: {tset} seed {seed} has no predictions of {', '.join(absent)}; its Holm family shrinks "
                  f"to {len(baselines) - len(absent)} of {len(baselines)} baselines x {len(METRICS)} metrics")
    groups = {t: load_groups(cfg, t) for t in GROUPED_TEST_SETS if any(k[0] == t for k in preds)}
    rng = np.random.default_rng(a.seed)

    rows: List[Dict] = []
    by_pair = defaultdict(list)
    for (tset, aug, seed), ref in sorted(preds.items()):
        if aug != a.reference:
            continue
        strata = group_strata(ref["names"], ref["y_true"], groups[tset]) if groups.get(tset) else None
        for (tset_b, aug_b, seed_b), base in sorted(preds.items()):
            if tset_b != tset or seed_b != seed or aug_b not in baselines:
                continue
            same = (ref["names"] == base["names"]) if ref["names"] and base["names"] else True
            if not same or not np.array_equal(ref["y_true"], base["y_true"]):
                raise RuntimeError(f"Test sets differ between {aug}/{aug_b} on {tset} seed {seed}")
            boot = paired_bootstrap(ref, base, a.n_boot, rng, strata)
            unit = GROUPED_TEST_SETS[tset][1] if strata else "image"
            row = {"test_set": tset, "seed": seed, "baseline": aug_b, "unit": unit,
                   "n_units": sum(len(s) for s in strata) if strata else int(ref["y_true"].size)}
            for name in METRICS:
                observed, lo, hi, p_value = boot[name]
                row[f"delta_{name}"] = round(observed, 4)
                row[f"{name}_ci95_low"] = round(lo, 4)
                row[f"{name}_ci95_high"] = round(hi, 4)
                row[f"{name}_p"] = p_value
                by_pair[(tset, aug_b, name)].append(observed)
            rows.append(row)
            parts = " | ".join(f"{n} {boot[n][0]:+.4f} [{boot[n][1]:+.4f}, {boot[n][2]:+.4f}]" for n in METRICS)
            print(f"{tset:8s} seed {seed}  {a.reference} - {aug_b:12s}  {parts}")
    if not rows:
        raise SystemExit(f"No {a.reference} run shares a test set and seed with a baseline.")
    _apply_holm(rows)

    # one seed: no spread to report
    summary = [{
        "test_set": tset, "baseline": base, "metric": metric, "seeds": len(v),
        "mean_delta": round(float(np.mean(v)), 4),
        "std_delta": round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else "n/a (1 seed)",
        "positive_every_seed": ("yes" if all(x > 0 for x in v) else "no") if len(v) > 1 else "n/a",
    } for (tset, base, metric), v in sorted(by_pair.items())]

    export_results_table(rows, _tables(cfg, "statistical_tests", a.rescore),
                         caption=(f"Paired bootstrap ({a.n_boot:,} resamples), {a.reference} minus baseline: "
                                  f"delta is the difference on the full test set, the 95 % CI is the bootstrap "
                                  f"percentile interval and p is floored at 1/(B+1). "
                                  f"Whole groups are resampled within class (column unit): Aria repeated captures of "
                                  f"one smear field, ALL-IDB2 crops of one cell, since neither are independent images. "
                                  f"Significant = "
                                  f"Holm-adjusted two-sided bootstrap p < 0.05 within each test set and seed (all "
                                  f"baselines x metrics); CIs and raw p-values are in the JSON. MorphoMix alone was "
                                  f"tuned on the test cohorts (exploratory), so a test-set lead is not a fair "
                                  f"comparison."),
                         label="tab:stats",
                         columns=["test_set", "seed", "baseline", "unit", "n_units", "delta_macro_f1",
                                  "macro_f1_significant", "delta_roc_auc", "roc_auc_significant", "delta_auprc",
                                  "auprc_significant"])
    if any(len(v) > 1 for v in by_pair.values()):
        export_results_table(
            summary, _tables(cfg, "statistical_tests_summary", a.rescore),
            caption=("Across-seed mean and sd of the MorphoMix minus baseline difference on the full test set; "
                     "positive_every_seed says whether it is positive in every seed (descriptive, not a test)."),
            label="tab:stats_summary")
    else:
        print("  every pair has a single seed - the across-seed summary table is not written "
              "(a std over one value is not a spread; use the bootstrap CIs instead)")
    print(f"done: {_tables(cfg, 'statistical_tests', a.rescore)}*")


CKPT_PATTERN = re.compile(r"phase1_(?P<arm>.+)_seed(?P<seed>\d+)_best\.pt$")


def phase1_checkpoints(cfg: Dict, screen: bool = False) -> List[tuple]:
    """Checkpoints of the protocol, or with `screen` also those of val-only screening runs."""
    dirs = [cfg["results_dir"]] + ([f"{cfg['results_dir']}/screen"] if screen else [])
    paths = [p for d in dirs for p in glob.glob(os.path.join(abs_path(d), "checkpoints", "phase1_*_best.pt"))]
    found = []
    for path in sorted(paths):
        m = CKPT_PATTERN.search(os.path.basename(path))
        if m:
            found.append((m["arm"], int(m["seed"]), path))
    return found


def _summarise(rows: List[Dict], metric_cols: List[str]) -> List[Dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[r["arm"]].append(r)
    out = []
    for arm, rs in groups.items():
        row = {"arm": arm, "seeds": len(rs)}
        for c in metric_cols:
            v = np.array([r[c] for r in rs], dtype=float)
            row[c] = f"{v.mean():.3f} +/- {v.std(ddof=1):.3f}" if v.size > 1 else f"{v.mean():.3f}"
        out.append(row)
    return out


def load_val_cohort(cfg: Dict, limit: int = 0) -> tuple:
    """The LeukemiaAttri val cohort in memory (`load_cohort`), for the val-only stress and screening scores."""
    val_dir = os.path.join(abs_path(cfg["phase1"]["data_dir"]), VAL_DIR)
    return load_cohort(BinaryLeukemiaDataset(val_dir, get_val_transforms(cfg["img_size"]), limit=limit))


def stress_main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Trained checkpoints on the val stress suite; no test cohort is loaded.")
    p.add_argument("--views", type=int, default=None, help="TTA views (default: config tta_views)")
    p.add_argument("--only", nargs="+", default=None, metavar="ID",
                   help="experiment ids to evaluate, e.g. phase1_morpho_mix_seed42")
    p.add_argument("--selection-only", action="store_true",
                   help="screening: also the val-only screening checkpoints, with the share of CAM energy in the "
                        "cell (stress_screen table)")
    p.add_argument("--limit", type=int, default=0, help="truncate the val cohort, class-balanced (smoke test)")
    a = p.parse_args(argv)
    cfg = load_config()
    if a.limit:
        refuse_real_results_dir(cfg["results_dir"])
    console = get_console()
    views = a.views or int(cfg.get("tta_views", 8))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val = load_val_cohort(cfg, a.limit)
    ckpts = phase1_checkpoints(cfg, screen=a.selection_only)
    if a.only:
        ckpts = [c for c in ckpts if f"phase1_{c[0]}_seed{c[1]}" in a.only]
    if not ckpts:
        console.print(f"[warning]No matching checkpoints under {cfg['results_dir']}.[/warning]")
        return 1
    if a.selection_only:
        screen_on_val(cfg, ckpts, val, views, device, console)
        return 0
    console.print(f"{len(ckpts)} checkpoints | val {len(val[1])} | TTA {views}")

    rows = []
    for arm, seed, path in ckpts:
        model = load_phase1_model(cfg, path, device)
        s = evaluate_families(model, val[0], val[1], ["clean"] + SELECTION, tta_views=views)
        del model
        torch.cuda.empty_cache()
        row = {"arm": arm, "seed": seed, "J_selection": selection_score(s), "clean_auc": s["clean"]["roc_auc"]}
        for f in SELECTION:
            row[f"{f}_auc"], row[f"{f}_pos"] = s[f]["roc_auc"], s[f]["pos_rate"]
        rows.append({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
        console.print(f"  {arm:>16s} seed {seed}: J_selection {row['J_selection']:.4f} | "
                      f"clean AUC {row['clean_auc']:.4f}")

    export_results_table(rows, _tables(cfg, "stress_selection_val"), caption=(
        f"Selection stress families on the LeukemiaAttri val cohort (TTA {views}), the cohort checkpoints are "
        "selected on: J (0.5 ROC-AUC + 0.5 macro-F1, mean over families) and ROC-AUC per family. Positive rates are "
        "in the JSON."), label="tab:stress_sel",
        columns=["arm", "seed", "J_selection", "clean_auc"] + [f"{f}_auc" for f in SELECTION])
    export_results_table(_summarise(rows, ["J_selection", "clean_auc"]), _tables(cfg, "stress_selection_val_summary"),
                         caption="Selection stress suite on the val cohort, mean +/- sd over seeds.",
                         label="tab:stress_sel_sum")
    console.print(f"ok tables in {_tables(cfg, '')}")
    return 0


def _cam_in_cell(extractor, images_u8: torch.Tensor, labels: np.ndarray, cells: torch.Tensor, device: str,
                 batch_size: int = 32) -> float:
    """Mean share of CAM energy inside the cell mask, for the true class."""
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    shares = []
    for b in range(0, len(images_u8), batch_size):
        x = (images_u8[b:b + batch_size].to(device).float() / 255.0 - mean) / std
        y = torch.as_tensor(labels[b:b + batch_size], device=device)
        cam = extractor.generate(x, y)[:, 0]
        m = cells[b:b + batch_size].to(device)
        shares.append(((cam * m).flatten(1).sum(1) / cam.flatten(1).sum(1).clamp(min=1e-8)).cpu())
    return float(torch.cat(shares).mean())


def val_cell_masks(cfg: Dict, val: tuple) -> torch.Tensor:
    """The XAI pseudo ground truth of every val image: the Azure-B mask cut to the central cell."""
    tau = float(cfg["morpho_threshold"])
    return torch.stack([torch.from_numpy(central_cell_mask(im.permute(1, 2, 0).numpy(), tau)) for im in val[0]])


def screen_row(cfg: Dict, arm: str, seed: int, path: str, val: tuple, cells: torch.Tensor, views: int, device: str,
               console) -> Dict:
    """One checkpoint on the val cohort: clean val, selection families, CAM energy in the cell (`val_cell_masks`)."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = load_phase1_model(cfg, path, device)
    s = evaluate_families(model, val[0], val[1], ["clean"] + SELECTION, tta_views=views)
    evidence_in_cell = _cam_in_cell(EvidenceCAM(model), val[0], val[1], cells, device)
    lcam = LayerCAM(model, get_target_cam_layer(model, stage_idx=-1))
    layercam_in_cell = _cam_in_cell(lcam, val[0], val[1], cells, device)
    lcam.remove_hooks()
    del model
    torch.cuda.empty_cache()
    row = {"id": f"phase1_{arm}_seed{seed}", "val_macro_f1": round(float(state["macro_f1"]), 4),
           "best_epoch": state.get("epoch"), "J_selection": round(selection_score(s), 4),
           "clean_auc": round(s["clean"]["roc_auc"], 4), "clean_f1": round(s["clean"]["macro_f1"], 4)}
    row.update({"evidence_in_cell": round(evidence_in_cell, 4), "layercam4_in_cell": round(layercam_in_cell, 4)})
    row.update({f"{f}_auc": round(s[f]["roc_auc"], 4) for f in SELECTION})
    console.print(f"  {row['id']:<32} val F1 {row['val_macro_f1']:.4f} | J_selection {row['J_selection']:.4f} | "
                  f"clean AUC {row['clean_auc']:.4f} | CAM in cell: evidence {evidence_in_cell:.3f}, "
                  f"Layer-CAM st4 {layercam_in_cell:.3f}")
    return row


def screen_on_val(cfg: Dict, ckpts: List[tuple], val: tuple, views: int, device: str, console) -> List[Dict]:
    """Val-only comparison of candidate configurations: clean val, selection families, CAM energy in cells."""
    cells = val_cell_masks(cfg, val)
    rows = [screen_row(cfg, arm, seed, path, val, cells, views, device, console) for arm, seed, path in ckpts]
    export_results_table(rows, _tables(cfg, "stress_screen"),
                         caption=f"Screening on the LeukemiaAttri val cohort only (TTA {views}): checkpoint val "
                                 f"macro-F1, J over the selection stress families, and the share of CAM energy inside "
                                 f"the central cell (the Azure-B mask cut to the centre cell; partly circular for "
                                 f"MorphoMix, which trains on the same mask). No test cohort is evaluated.",
                         label="tab:stress_screen",
                         columns=["id", "val_macro_f1", "J_selection", "clean_auc", "evidence_in_cell",
                                  "layercam4_in_cell", "best_epoch"])
    return rows


def profile_main(argv=None):
    cfg = load_config()
    p = argparse.ArgumentParser(description="Backbone parameters, GFLOPs and batch-1 latency.")
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--cpu-only", action="store_true")
    a = p.parse_args(argv)

    model = build_model(cfg["model_name"], num_classes=len(cfg["phase1"]["class_names"]),
                        pretrained=False, dropout=cfg["dropout"])
    shape = (1, 3, cfg["img_size"], cfg["img_size"])
    prof = profile_model(model, input_shape=shape)
    row = {"model": cfg["model_name"], "input": f"{shape[2]}x{shape[3]}",
           "params": prof["params"], "GMACs": prof["macs"], "GFLOPs": prof["flops"], "size_fp32": prof["size_mb"]}
    devices = ["cpu"] + ([] if a.cpu_only or not torch.cuda.is_available() else ["cuda"])
    for dev in devices:
        lat = benchmark_inference_latency(model, input_shape=shape, device=dev, iterations=a.iterations)
        row[f"latency_{dev}"] = lat["latency_str"]
        row[f"fps_{dev}"] = lat["fps_str"]
    for k, v in row.items():
        print(f"{k:14s} {v}")
    export_results_table([row], _tables(cfg, "model_profile"),
                         caption="Backbone size and batch-1 inference latency.", label="tab:profile")


# --- `xai`: CAM localisation ---

XAI_DOC = """Quantitative XAI localisation and CAM figures of the trained models on the ALL-IDB2 single cells."""
METRIC_KEYS = ["pointing_game_hit_rate_pct", "energy_in_mask_pct", "mean_iou_pct", "avg_drop_pct", "increase_conf_pct"]
N_STAGES = 4  # ConvNeXt V2 Atto; stage_idx -1 is stage 4


def pseudo_gt_masks(images: torch.Tensor, threshold: float) -> torch.Tensor:
    return torch.stack([torch.from_numpy(central_cell_mask(denormalize_image(im), threshold)) for im in images])


def run_metrics(model, extractor, loader, threshold: float, device) -> Dict[str, float]:
    """Image-count-weighted mean of evaluate_xai_metrics over the loader."""
    acc = {k: 0.0 for k in METRIC_KEYS}
    n_total = 0
    for images, targets, _ in loader:
        images, targets = images.to(device), targets.to(device)
        cams = extractor.generate(images, targets)
        masks = pseudo_gt_masks(images.cpu(), threshold).to(device)
        m = evaluate_xai_metrics(model, images, targets, cams, cell_masks=masks, device=device)
        n = images.shape[0]
        for k in METRIC_KEYS:
            acc[k] += m[k] * n
        n_total += n
    return {k: acc[k] / max(1, n_total) for k in METRIC_KEYS}


def one_image_per_class(ds: BinaryLeukemiaDataset, n_classes: int) -> List[int]:
    picks = {}
    for idx, (_, y) in enumerate(ds.samples):
        picks.setdefault(y, idx)
    return [picks[c] for c in range(n_classes) if c in picks]


def xai_main(argv=None):
    cfg = load_config()
    p = argparse.ArgumentParser(description=XAI_DOC, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=cfg["seeds"][0])
    p.add_argument("--arms", nargs="+", default=cfg["augmentations"])
    p.add_argument("--limit", type=int, default=0, help="evaluate only N test images, class-balanced (smoke test)")
    a = p.parse_args(argv)
    if a.limit:
        refuse_real_results_dir(cfg["results_dir"])

    console = get_console()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = cfg["phase1"]["class_names"]
    tau = float(cfg["morpho_threshold"])
    ckpts = {aug: os.path.join(abs_path(cfg["results_dir"]), "checkpoints", f"phase1_{aug}_seed{a.seed}_best.pt")
             for aug in a.arms}
    missing = [c for c in ckpts.values() if not os.path.exists(c)]
    if missing:
        raise FileNotFoundError(f"Checkpoints missing, e.g. {missing[0]}; run `main.py pretrain` first.")
    test_ds = BinaryLeukemiaDataset(os.path.join(abs_path(cfg["phase1"]["data_dir"]), TEST_SETS["allidb2"]),
                                    get_val_transforms(cfg["img_size"]), limit=a.limit)
    loader = DataLoader(test_ds, batch_size=min(16, cfg["batch_size"]), shuffle=False)
    console.print(f"[dim]XAI evaluation on {len(test_ds)} ALL-IDB2 images, seed {a.seed}, device {device}[/dim]")

    # Sample images for figures (one per class)
    pick = one_image_per_class(test_ds, len(names))
    sample_x = torch.stack([test_ds[i][0] for i in pick])
    sample_imgs = [denormalize_image(x) for x in sample_x]
    sample_y = torch.tensor([test_ds.samples[i][1] for i in pick], device=device)
    sample_x = sample_x.to(device)
    gt_masks = [central_cell_mask(im, tau) for im in sample_imgs]
    row_labels = [names[test_ds.samples[i][1]] for i in pick]

    # Layer-CAM at every stage in xai_stages plus the exact evidence map (all reported, none picked)
    stages = [int(s) for s in cfg["xai_stages"]]
    methods = [(f"Layer-CAM stage {N_STAGES + s + 1}", f"stage{N_STAGES + s + 1}", s) for s in stages]
    methods.append(("Evidence map stage 4", "evidence", None))
    rows = []
    grids = {tag: [[draw_contour(im, gm)] for im, gm in zip(sample_imgs, gt_masks)] for _, tag, _ in methods}
    for aug in a.arms:
        model = load_phase1_model(cfg, ckpts[aug], device)
        for title, tag, s in methods:
            ext = EvidenceCAM(model) if s is None else LayerCAM(model, get_target_cam_layer(model, stage_idx=s))
            m = run_metrics(model, ext, loader, tau, device)
            rows.append({"augmentation": aug, "cam": title, **{k: round(v, 2) for k, v in m.items()}})
            console.print(f"  {aug:12s} {title:22s} PG {m['pointing_game_hit_rate_pct']:.1f}%  "
                          f"energy {m['energy_in_mask_pct']:.1f}%  IoU {m['mean_iou_pct']:.1f}%")
            cams = ext.generate(sample_x, sample_y)[:, 0].cpu().numpy()
            for r in range(len(pick)):
                grids[tag][r].append(overlay_heatmap(sample_imgs[r], cams[r]))
            ext.remove_hooks()
        del model
        torch.cuda.empty_cache()
    export_results_table(rows, abs_path(os.path.join(cfg["results_dir"], "tables", "xai_localization_arms")),
                         caption=(f"Localisation of the models (seed {a.seed}) on the ALL-IDB2 single cells, "
                                  f"for the true class: Layer-CAM at every stage and the exact evidence map, all "
                                  f"reported. The pseudo ground truth is the Azure-B cell mask (tau = {tau:.2f}) cut "
                                  f"to the cell at the image centre, i.e. the mask MorphoMix trains on, so the in-cell "
                                  f"scores are partly circular for morpho_mix."),
                         label="tab:xai_arms")
    arm_titles = [figstyle.ARM_LABEL.get(x, x) for x in a.arms]
    fig_dir = os.path.join(abs_path(cfg["results_dir"]), "figures", "cam_comparison")
    for title, tag, _ in methods:
        render_panel_grid(grids[tag], ["Image + pseudo-GT"] + arm_titles, row_labels,
                          os.path.join(fig_dir, f"all_models_cam_comparison_{tag}.png"), suptitle=title)

    console.print("[success]done[/success] XAI evaluation")


# --- `analysis`: extended metrics of the reported runs ---

ANALYSIS_DOC = """Extended metrics of every run from its stored predictions (no model is loaded): ROC-AUC with a 95 %
bootstrap CI (resampling groups within class: Aria field groups, ALL-IDB2 cell groups, val slides), AUPRC and its
prevalence floor, calibration (Brier score, ECE over 10 equal-width bins), sensitivity at 90 / 95 % specificity
(points of the ROC curve, no threshold), and at the primary decision rule balanced accuracy, sensitivity, specificity,
MCC and macro-F1. Also Aria ROC-AUC per B-ALL subtype against Benign, the seed ensemble of every arm (mean p(ALL)) and
selection-free val AUC from the training log (mean over epochs, epoch 5, last epoch)."""
ECE_BINS = 10
FOUNDATION_ARMS = ("dinobloom_s", "dinobloom_b")  # scripts/foundation.py MODELS, one frozen-probe run each
PRED_FILE = re.compile(r"^phase1_(?P<aug>[a-z_0-9]+?)_seed(?P<seed>\d+)_(?:test_)?(?P<cohort>val|allidb2|aria)\.json$")
SUMMARY_COLUMNS = ["cohort", "arm", "seeds", "roc_auc", "auprc", "sens_at_spec90", "balanced_acc", "sensitivity",
                   "specificity", "mcc", "macro_f1", "brier", "ece"]


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = ECE_BINS) -> float:
    """|accuracy - confidence| of p(ALL) per equal-width bin, weighted by the bin's share of the images."""
    which = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    return float(sum(abs(y[which == b].mean() - p[which == b].mean()) * (which == b).mean()
                     for b in range(bins) if (which == b).any()))


def sensitivity_at_specificity(y: np.ndarray, p: np.ndarray, specificity: float) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    # 1 - 0.9 is 0.0999...; the tolerance keeps the ROC point at exactly that FPR (e.g. 13/130)
    return float(tpr[fpr <= 1 - specificity + 1e-9].max())


def unit_members(y: np.ndarray, groups: Optional[np.ndarray]) -> List[List[np.ndarray]]:
    """Per class, the image indices of each resampling unit (a group, or an image when there are no groups)."""
    units = groups if groups is not None else np.arange(len(y))
    return [[np.flatnonzero((units == u) & (y == c)) for u in np.unique(units[y == c])] for c in (0, 1)]


def resample_units(by_class: List[List[np.ndarray]], rng: np.random.Generator) -> np.ndarray:
    return np.concatenate([np.concatenate([cls[i] for i in rng.integers(0, len(cls), len(cls))]) for cls in by_class])


def grouped_auc_ci(y: np.ndarray, p: np.ndarray, groups: Optional[np.ndarray], n_boot: int,
                   rng: np.random.Generator) -> Tuple[float, float]:
    """95 % percentile CI of ROC-AUC, resampling groups (or images) with replacement within each class."""
    by_class = unit_members(y, groups)
    aucs = [roc_auc_score(y[idx], p[idx]) for idx in (resample_units(by_class, rng) for _ in range(n_boot))]
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


METHOD_METRICS = {"roc_auc": roc_auc_score, "auprc": average_precision_score}


def method_level_tests(cfg: Dict, preds: Dict[Tuple[str, str, int], Dict], reference: str, n_boot: int,
                       rng: np.random.Generator, groups_fn=None) -> List[Dict]:
    """Reference minus comparator of the seed-mean metric, with a hierarchical bootstrap that resamples the units
    (groups within class, shared by every arm, since all score the same images) and, independently per arm, the
    seeds (each training run is one draw of the method). Two-sided p floored at 1/(B+1), Holm within each (cohort,
    metric) family over the comparators. A single-run method (a frozen probe) contributes its one run. `groups_fn`
    (cfg, cohort, names) -> units replaces `analysis_groups` for a cohort outside the protocol."""
    groups_fn = groups_fn or analysis_groups
    rows = []
    for cohort in sorted({k[0] for k in preds}):
        arms = sorted({k[1] for k in preds if k[0] == cohort})
        if reference not in arms:
            continue
        seeds = {a: sorted(k[2] for k in preds if k[0] == cohort and k[1] == a) for a in arms}
        first = preds[(cohort, reference, seeds[reference][0])]
        y, names = np.asarray(first["y_true"]), [os.path.basename(n) for n in first["names"]]
        pos = {n: i for i, n in enumerate(names)}
        order = {a: {s: [pos[os.path.basename(n)] for n in preds[(cohort, a, s)]["names"]] for s in seeds[a]}
                 for a in arms}
        prob = {}
        for a in arms:
            for s in seeds[a]:
                run = preds[(cohort, a, s)]
                if len(order[a][s]) != len(y) or not np.array_equal(np.asarray(run["y_true"]), y[order[a][s]]):
                    raise RuntimeError(f"{cohort} {a} seed {s}: images or labels differ from the reference run")
                p_run = np.empty(len(y))
                p_run[order[a][s]] = np.asarray(run["probs"], dtype=np.float64)[:, 1]
                prob[(a, s)] = p_run
        by_class = unit_members(y, groups_fn(cfg, cohort, names))
        comparators = [a for a in arms if a != reference]
        for metric, fn in METHOD_METRICS.items():
            observed = {a: np.mean([fn(y, prob[(a, s)]) for s in seeds[a]]) for a in arms}
            deltas = {a: [] for a in comparators}
            for _ in range(n_boot):
                idx = resample_units(by_class, rng)
                draw = {a: np.mean([fn(y[idx], prob[(a, s)][idx]) for s in rng.choice(seeds[a], len(seeds[a]))])
                        for a in arms}
                for a in comparators:
                    deltas[a].append(draw[reference] - draw[a])
            family = []
            for a in comparators:
                d = np.asarray(deltas[a])
                p_two = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
                family.append({"cohort": cohort, "metric": metric, "comparator": a,
                               "reference_seeds": len(seeds[reference]), "comparator_seeds": len(seeds[a]),
                               "delta": round(observed[reference] - observed[a], 4),
                               "ci_low": round(float(np.percentile(d, 2.5)), 4),
                               "ci_high": round(float(np.percentile(d, 97.5)), 4),
                               "p": max(p_two, 1.0 / (n_boot + 1))})
            for r, adj in zip(family, holm([r["p"] for r in family])):
                r["p_holm"] = round(adj, 4)
                r["significant"] = "yes" if adj < 0.05 else "no"
            rows += family
    return rows


def cohort_metrics(y: np.ndarray, p: np.ndarray, threshold: float, groups: Optional[np.ndarray], n_boot: int,
                   rng: np.random.Generator) -> Dict:
    pred = (p >= threshold).astype(int)
    lo, hi = grouped_auc_ci(y, p, groups, n_boot, rng)
    return {"roc_auc": roc_auc_score(y, p), "auc_ci_low": lo, "auc_ci_high": hi,
            "auprc": average_precision_score(y, p), "prevalence": float(y.mean()),
            "brier": brier_score_loss(y, p), "ece": expected_calibration_error(y, p),
            "sens_at_spec90": sensitivity_at_specificity(y, p, 0.90),
            "sens_at_spec95": sensitivity_at_specificity(y, p, 0.95), "threshold": threshold,
            "balanced_acc": balanced_accuracy_score(y, pred), "sensitivity": float(pred[y == 1].mean()),
            "specificity": float(1 - pred[y == 0].mean()), "mcc": matthews_corrcoef(y, pred),
            "macro_f1": f1_score(y, pred, average="macro"), "predicted_positive_rate": float(pred.mean())}


def analysis_groups(cfg: Dict, cohort: str, names: List[str]) -> Optional[np.ndarray]:
    """Resampling units of a cohort: test-set groups from `split`, val slides from the val manifest."""
    if cohort == "val":
        path = os.path.join(abs_path(cfg["phase1"]["data_dir"]), "val", "manifest.csv")
        with open(path, encoding="utf-8", newline="") as f:
            slide = {os.path.basename(row["file"]): row["slide"] for row in csv.DictReader(f)}
        return np.array([slide[os.path.basename(n)] for n in names])
    groups = load_groups(cfg, cohort)
    return None if groups is None else np.array([groups[os.path.basename(n)] for n in names])


def group_runs(runs: List[Dict]) -> Dict[Tuple[str, str], List[Dict]]:
    groups: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for r in runs:
        groups[(r["cohort"], r["arm"])].append(r)
    return dict(sorted(groups.items()))


def mean_sd(values: List[float]) -> str:
    return f"{np.mean(values):.3f} +/- {np.std(values, ddof=1):.3f}" if len(values) > 1 else f"{values[0]:.3f}"


def selection_free_val(cfg: Dict) -> List[Dict]:
    """Val ROC-AUC per phase-1 arm from the logged curves: the selected maximum and selection-free summaries."""
    path = os.path.join(abs_path(cfg["results_dir"]), "logs", "training_log.csv")
    curves = defaultdict(list)
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if re.match(r"^phase1_[a-z_]+_seed\d+$", row["experiment_id"]):
                curves[row["experiment_id"]].append((int(row["epoch"]), float(row["val_roc_auc"])))
    per_arm = defaultdict(list)
    for eid, c in curves.items():
        v = np.array([x for _, x in sorted(dict(c).items())])  # a resumed or duplicated run logs an epoch twice
        per_arm[eid.rsplit("_seed", 1)[0].replace("phase1_", "")].append(
            {"selected": v.max(), "mean_all_epochs": v.mean(), "mean_epochs_1_10": v[:10].mean(),
             "epoch_5": dict(c).get(5, np.nan), "last_epoch": v[-1]})
    return [{"arm": arm, "seeds": len(vals), **{k: mean_sd([x[k] for x in vals]) for k in vals[0]}}
            for arm, vals in per_arm.items()]


def analysis_main(argv=None):
    cfg = load_config()
    p = argparse.ArgumentParser(description=ANALYSIS_DOC)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    a = p.parse_args(argv)
    console = get_console()
    rng = np.random.default_rng(a.seed)
    pred_dir = os.path.join(abs_path(cfg["results_dir"]), "predictions")
    out = os.path.join(abs_path(cfg["results_dir"]), "tables")
    rule = cfg["inference"]["decision_threshold"]
    runs = []
    for path in sorted(glob.glob(os.path.join(pred_dir, "phase1_*_seed*_*.json"))):
        m = PRED_FILE.match(os.path.basename(path))
        if not m:
            continue
        d = _load_json(path)
        y, pp = np.asarray(d["y_true"]), np.asarray(d["probs"], dtype=np.float64)[:, 1]
        eid = f"phase1_{m['aug']}_seed{m['seed']}"
        t = rule_threshold(cfg, rule, eid, pp.tolist())
        row = {"cohort": m["cohort"], "arm": m["aug"], "seed": int(m["seed"]),
               **cohort_metrics(y, pp, t, analysis_groups(cfg, m["cohort"], d["names"]), a.n_boot, rng)}
        if m["cohort"] == "aria":
            subtype = np.array([os.path.basename(n).split("_")[1] for n in d["names"]])
            for s in ("Early", "Pre", "Pro"):
                k = (subtype == s) | (y == 0)
                row[f"auc_{s.lower()}_vs_benign"] = roc_auc_score(y[k], pp[k])
        runs.append(row)
        console.print(f"  {eid} {m['cohort']}: AUC {row['roc_auc']:.3f} [{row['auc_ci_low']:.3f}, "
                      f"{row['auc_ci_high']:.3f}] | bal acc {row['balanced_acc']:.3f} | ECE {row['ece']:.3f}")
    if not runs:
        raise SystemExit(f"no predictions under {pred_dir}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "extended_metrics_runs.json"), "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)

    summary = [{"cohort": cohort, "arm": arm, "seeds": len(rows),
                **{k: mean_sd([r[k] for r in rows]) for k in rows[0] if k not in ("cohort", "arm", "seed")}}
               for (cohort, arm), rows in group_runs(runs).items()]
    export_results_table(
        summary, os.path.join(out, "extended_metrics"),
        caption=("Extended metrics, mean +/- sd over seeds (per-run values and 95 % grouped-bootstrap CIs of ROC-AUC "
                 f"in extended_metrics_runs.json). Decision rule ALL iff p(ALL) >= {rule}; sensitivity at fixed "
                 "specificity and calibration need no threshold. Exploratory (see the unblinding record)."),
        label="tab:extended", columns=SUMMARY_COLUMNS)

    ensemble = []
    for (cohort, arm), rows in group_runs(runs).items():
        name = "val" if cohort == "val" else f"test_{cohort}"
        files = [os.path.join(pred_dir, f"phase1_{arm}_seed{r['seed']}_{name}.json") for r in rows]
        if len(files) < 2:
            continue
        avg = average_predictions([_load_json(f) for f in files])
        y, pp = np.asarray(avg["y_true"]), np.asarray(avg["probs"])[:, 1]
        t = float(rule) if not isinstance(rule, str) else 0.5
        m = cohort_metrics(y, pp, t, analysis_groups(cfg, cohort, avg["names"]), a.n_boot, rng)
        keep = ("roc_auc", "auc_ci_low", "auc_ci_high", "auprc", "balanced_acc", "macro_f1")
        ensemble.append({"cohort": cohort, "arm": arm, "seeds": len(files), **{k: round(m[k], 4) for k in keep}})
    if ensemble:
        export_results_table(ensemble, os.path.join(out, "seed_ensemble_metrics"),
                             caption="Seed ensemble per arm, mean p(ALL) over its seeds (exploratory).",
                             label="tab:ensemble")
    preds = {}
    for path in sorted(glob.glob(os.path.join(pred_dir, "phase1_*_seed*_*.json"))):
        m = PRED_FILE.match(os.path.basename(path))
        if m and (m["aug"] in cfg["augmentations"] or m["aug"] in FOUNDATION_ARMS):
            preds[(m["cohort"], m["aug"], int(m["seed"]))] = _load_json(path)
    method = method_level_tests(cfg, preds, "morpho_mix", a.n_boot, rng)
    if method:
        export_results_table(
            method, os.path.join(out, "method_level_tests"),
            caption=("MorphoMix minus each comparator, seed-mean ROC-AUC / AUPRC; 95 % CI and two-sided p of a "
                     f"hierarchical bootstrap ({a.n_boot:,} draws) over units (groups within class) and seeds; Holm "
                     "within each cohort and metric. Aria units are field groups, not patients (89 patients, no "
                     "patient IDs), so its intervals are optimistic. Exploratory (see the unblinding record); "
                     "MorphoMix alone was tuned on ALL-IDB2 and Aria, so its lead there is not a fair comparison."),
            label="tab:method_tests")
    if os.path.exists(os.path.join(abs_path(cfg["results_dir"]), "logs", "training_log.csv")):
        export_results_table(selection_free_val(cfg), os.path.join(out, "val_selection_free"),
                             caption=("Val ROC-AUC without checkpoint selection, from the logged val curves: the "
                                      "selected (maximum) value is optimistic, the others are selection-free."),
                             label="tab:val_free")
    console.print(f"[success]done[/success] extended metrics -> {out}")
