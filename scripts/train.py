"""Binary Normal vs ALL training: train C-NMC, select on LeukemiaAttri val, score the test cohorts once."""
import argparse
import gc
import glob
import hashlib
import json
import os
from typing import Dict, List, Tuple

from src.utils.config import load_config, abs_path, apply_overrides, refuse_real_results_dir, setup_cuda_env
setup_cuda_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.datasets.dataset import BinaryLeukemiaDataset  # noqa: E402
from src.datasets.loaders import make_loader, eval_loader  # noqa: E402
from src.augmentations.transforms import get_train_transforms, get_val_transforms  # noqa: E402
from src.augmentations.style_bank import StyleBank, bank_digest  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.training.engine import Trainer  # noqa: E402
from src.utils.seed import setup_run  # noqa: E402
from src.utils.exporter import export_results_table  # noqa: E402
from src.evaluation.metrics import threshold_free_metrics  # noqa: E402
from src.evaluation.calibration import DECISION_THRESHOLD, decision_metrics, resolve_threshold  # noqa: E402
from src.evaluation.field_inference import settings_summary  # noqa: E402
from src.evaluation import robust  # noqa: E402
from src.evaluation.field_inference import crop_scales  # noqa: E402
from src.evaluation.predict import field_inputs, predict_fields, predict_images  # noqa: E402
from src.evaluation.confusion import save_confusion  # noqa: E402
from src.utils.logger import (  # noqa: E402
    print_header_panel,
    print_ablation_header,
    print_master_comparison,
    get_console
)

TABLE_PREFIX = "tables/phase1_pretrain_results"  # under results_dir


# data/processed sub-folders: train and val are one laboratory each, every test cohort another one
TRAIN_DIR, VAL_DIR = "train", "val"
TEST_SETS = {"allidb2": "test_allidb2", "aria": "test_aria"}
TEST_SET_LABELS = {"allidb2": "ALL-IDB2 (single cell)", "aria": "Aria (full field)"}
MORPHO_ARMS = ("morpho_mix",)


def peak_lr(cfg: Dict) -> float:
    """Linear scaling rule of ConvNeXt V2 (Appendix A.2): lr = base_lr * effective batch / 256."""
    return cfg["phase1"]["base_lr"] * cfg["batch_size"] * cfg["grad_accum_steps"] / 256


def seeds_for(cfg: Dict, aug: str, requested: List[int]) -> List[int]:
    """Scheme B: primary seed for every arm, all seeds for headline arms."""
    primary = cfg.get("primary_seed", requested[0])
    if aug in cfg.get("headline_arms", []):
        return list(requested)
    return [primary] if primary in requested else [requested[0]]


def morpho_kwargs(cfg: Dict, aug: str) -> Dict:
    """MorphoMix component settings; C1 draws its references from the MLL23 style bank (`style_bank.path`), the
    RandStainNA hybrid (`rsn_prob`) its templates from the training-set fit (`randstainna_stats`)."""
    if aug not in MORPHO_ARMS:
        return {}
    refs = used_references(cfg, aug)
    bank = refs.get("style_bank")
    kwargs = {
        "style_bank": StyleBank.load(bank["path"]) if bank else None,
        "appearance_prob": float(cfg.get("appearance_prob", 0.0)),
        "appearance_alpha": tuple(cfg.get("appearance_alpha", (0.5, 1.0))),
        "virtual_template_prob": float(cfg.get("virtual_template_prob", 0.0)),
        "acquisition_prob": float(cfg.get("acquisition_prob", 0.0)),
        "rsn_prob": float(cfg.get("rsn_prob", 0.0)),
        "small_cell_prob": float(cfg.get("small_cell_prob", 0.0)),
        "small_cell_px": tuple(cfg.get("small_cell_px", (24, 48))),
        "small_cell_lowdetail_prob": float(cfg.get("small_cell_lowdetail_prob", 0.5)),
        "field_prob": float(cfg.get("field_prob", 0.0)),
        "field_cells": tuple(cfg.get("field_cells", (2, 8))),
        "use_background": bool(cfg.get("use_background", True)),
        "background_prob": float(cfg.get("background_prob", 0.5)),
        "background_rbc": tuple(cfg.get("background_rbc", (5, 25))),
        "field_rbc": tuple(cfg.get("field_rbc", (40, 120))),
        "feather_edges": bool(cfg.get("feather_edges", True)),
    }
    if "randstainna_stats" in refs:  # rsn_prob > 0
        kwargs["randstainna_stats"] = load_randstainna_stats(cfg)
        kwargs["randstainna_std_hyper"] = float(cfg["randstainna_std_hyper"])
    return kwargs


def recipe_kwargs(cfg: Dict) -> Dict:
    """Training-recipe settings shared by every arm; each default keeps the original recipe."""
    p1 = cfg["phase1"]
    return {
        "ema_decay": float(cfg.get("ema_decay", 0.0)),
        "label_smoothing": float(p1.get("label_smoothing", 0.0)),
        "freeze_stages": int(p1.get("freeze_stages", 0)),
        "freeze_epochs": int(p1.get("freeze_epochs", 0)),
        "lowres_prob": float(cfg.get("lowres_prob", 0.0)),
        "lowres_scale": tuple(cfg.get("lowres_scale", (0.25, 0.75))),
        "probe_lr": float(p1.get("probe_lr", 0.0)),
        "mixstyle_p": float(cfg.get("mixstyle_p", 0.0)),
        "mixstyle_alpha": float(cfg.get("mixstyle_alpha", 0.1)),
        "mixstyle_stages": tuple(cfg.get("mixstyle_stages", (0, 1))),
    }


def stain_kwargs(cfg: Dict, aug: str) -> Dict:
    """Stain-baseline settings (src/augmentations/stain_baselines.py); a missing stats file or bank is an error."""
    if aug not in ("hed_jitter", "randstainna", "stain_mixup"):
        return {}
    kwargs = {"stain_prob": float(cfg["stain_prob"]), "hed_sigma": float(cfg["hed_sigma"]),
              "randstainna_std_hyper": float(cfg["randstainna_std_hyper"])}
    if aug == "randstainna":
        kwargs["randstainna_stats"] = load_randstainna_stats(cfg)
    if aug == "stain_mixup":
        with np.load(abs_path(used_references(cfg, aug)["style_bank"]["path"])) as z:
            kwargs["stain_bank"] = torch.from_numpy(z["stain_matrix"])
    return kwargs


def load_randstainna_stats(cfg: Dict) -> Dict:
    """The RandStainNA training-set template distribution written by `prepare` (`fit_randstainna`)."""
    path = abs_path(cfg["randstainna_stats"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"RandStainNA stats {cfg['randstainna_stats']} not found: run `python main.py "
                                f"prepare` first")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def used_references(cfg: Dict, aug: str) -> Dict[str, Dict]:
    """The prepared artefacts an arm reads, {name: {path, sha256}}: the MLL23 style bank (morpho_mix with C1 on,
    stain_mixup; content digest), the RandStainNA training-set statistics (randstainna, morpho_mix with rsn_prob >
    0; file digest) and, with `aux_train.enabled`, the auxiliary training cells (every arm; digest of their aux.json,
    which holds the settings they were built with and their counts)."""
    refs = {}
    if aug == "stain_mixup" or (aug in MORPHO_ARMS and float(cfg.get("appearance_prob", 0.0)) > 0):
        path = cfg["style_bank"]["path"]
        if not os.path.exists(abs_path(path)):
            raise FileNotFoundError(f"style bank {path} not found: build it with `python main.py prepare` (needs "
                                    f"data/raw/MLL23), or pass --set appearance_prob=0 (no MLL23 form of C1)")
        refs["style_bank"] = {"path": path, "sha256": bank_digest(path)}
    if aug == "randstainna" or (aug in MORPHO_ARMS and float(cfg.get("rsn_prob", 0.0)) > 0):
        path = cfg["randstainna_stats"]
        if not os.path.exists(abs_path(path)):
            raise FileNotFoundError(f"RandStainNA stats {path} not found: run `python main.py prepare` first")
        with open(abs_path(path), "rb") as f:
            refs["randstainna_stats"] = {"path": path, "sha256": hashlib.sha256(f.read()).hexdigest()}
    if cfg["aux_train"]["enabled"]:
        path = f"{cfg['aux_train']['dir']}/aux.json"
        if not os.path.exists(abs_path(path)):
            raise FileNotFoundError(f"auxiliary cells {path} not found: run `python main.py prepare` (needs "
                                    f"data/raw/Bodzas2023)")
        with open(abs_path(path), "rb") as f:
            raw = f.read()
        counts = json.loads(raw)["counts"]
        found = {c: len(glob.glob(abs_path(f"{cfg['aux_train']['dir']}/{c}/*.png"))) for c in counts}
        if found != counts:
            raise RuntimeError(f"auxiliary cells: {found} images on disk, aux.json lists {counts}; rebuild with "
                               f"`python main.py prepare`")
        refs["aux_train"] = {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}
    return refs


def check_references(cfg: Dict, aug: str, experiment_id: str) -> None:
    """A finished run is reused (resume) only when it read the same prepared artefacts (`used_references`) as the
    config gives now."""
    path = abs_path(f"{cfg['results_dir']}/checkpoints/{experiment_id}_best.pt")
    if not os.path.exists(path):
        return
    used = torch.load(path, map_location="cpu", weights_only=False).get("references", {})
    now = used_references(cfg, aug)
    if {k: v["sha256"] for k, v in used.items()} != {k: v["sha256"] for k, v in now.items()}:
        raise RuntimeError(f"{experiment_id} was trained with {used}, but the config now gives {now}; move the old "
                           f"run away or use a new results_dir")


def build_loaders(cfg: Dict, seed: int, limit: int = 0, aug: str = "basic", with_tests: bool = True):
    """Datasets and loaders for train, val and, optionally, the test cohorts."""
    p1 = cfg["phase1"]
    img = cfg["img_size"]
    root = abs_path(p1["data_dir"])
    with_prior = aug in MORPHO_ARMS  # the Azure-B prior is computed in the loader workers
    extra = [abs_path(cfg["aux_train"]["dir"])] if cfg["aux_train"]["enabled"] else []
    datasets = {
        "train": BinaryLeukemiaDataset(os.path.join(root, TRAIN_DIR), get_train_transforms(img), limit=limit,
                                       with_prior=with_prior, fraction=float(p1.get("train_fraction", 1.0)),
                                       subset_seed=int(p1.get("train_subset_seed", 0)), extra_dirs=extra),
        "val": BinaryLeukemiaDataset(os.path.join(root, VAL_DIR), get_val_transforms(img),
                                     limit=limit),
    }
    for key, sub in (TEST_SETS.items() if with_tests else ()):
        datasets[key] = BinaryLeukemiaDataset(os.path.join(root, sub), get_val_transforms(img),
                                              limit=limit)
    for k, ds in datasets.items():
        if len(ds) == 0:
            raise RuntimeError(f"Empty split '{k}' under {root}; run `python main.py prepare` first.")

    loaders = {"train": make_loader(datasets["train"], cfg, shuffle=True, seed=seed, drop_last=True)}
    for k in ["val"] + (list(TEST_SETS) if with_tests else []):
        loaders[k] = eval_loader(datasets[k], cfg)
    return datasets, loaders


def class_counts(ds: BinaryLeukemiaDataset, n_classes: int) -> List[int]:
    counts = [0] * n_classes
    for _, y in ds.samples:
        counts[y] += 1
    return counts


def inverse_frequency_weights(counts: List[int]) -> torch.Tensor:
    tot = float(sum(counts))
    k = len(counts)
    return torch.tensor([tot / (k * max(1, c)) for c in counts], dtype=torch.float32)


def run_one(cfg: Dict, aug: str, seed: int, epochs: int, console, limit: int = 0, tag: str = None,
            with_tests: bool = True) -> Dict:
    """`tag` overrides the arm name in experiment_id (ablation and screening runs)."""
    # a run that failed under `all --continue-on-error` leaves its trainer (CUDA tensors, loader workers) in
    # reference cycles through the traceback
    gc.collect()
    torch.cuda.empty_cache()
    setup_run(cfg, seed)
    p1 = cfg["phase1"]
    class_names = p1["class_names"]
    experiment_id = f"phase1_{tag or aug}_seed{seed}"

    datasets, loaders = build_loaders(cfg, seed, limit, aug=aug, with_tests=with_tests)
    references = used_references(cfg, aug)
    counts = class_counts(datasets["train"], len(class_names))
    weights = inverse_frequency_weights(counts)

    model = build_model(
        model_name=cfg["model_name"],
        num_classes=len(class_names),
        pretrained=cfg["pretrained"],
        dropout=cfg["dropout"],
        drop_path=cfg["drop_path"],
        head_init_scale=cfg["head_init_scale"],
    )

    trainer = Trainer(
        model=model,
        model_name=cfg["model_name"],
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        class_names=class_names,
        augmentation=aug,
        morpho_threshold=cfg["morpho_threshold"],
        tta_views=cfg.get("tta_views", 8),
        selection_metric=p1.get("selection_metric", "macro_f1"),
        **recipe_kwargs(cfg),
        **morpho_kwargs(cfg, aug),
        **stain_kwargs(cfg, aug),
        epochs=epochs,
        warmup_epochs=p1["warmup_epochs"],
        lr=peak_lr(cfg),
        layer_decay=p1["layer_decay"],
        weight_decay=p1["weight_decay"],
        grad_accum_steps=cfg["grad_accum_steps"],
        mixed_precision=cfg["mixed_precision"],
        channels_last=cfg["channels_last"],
        log_every=cfg["log_every"],
        experiment_id=experiment_id,
        seed=seed,
        output_dir=cfg["results_dir"],
        resources=cfg.get("resources"),
        class_weights=weights
    )

    result = trainer.run()
    if references:  # the prepared artefacts this run read, checked on resume by `check_references`
        state = torch.load(trainer.ckpt_path, map_location="cpu", weights_only=False)
        state["references"] = result["references"] = references
        torch.save(state, trainer.ckpt_path)

    # score the selected checkpoint on both test cohorts
    trainer.load_best()
    summary = []
    if with_tests:
        scores, summary = score_tests(cfg, experiment_id, trainer.model, trainer.accelerator.device, datasets, loaders,
                                      autocast=trainer.accelerator.autocast)
        result.update(scores)
    console.print(
        f"[bold]{experiment_id}[/bold] val macro-F1 {result['macro_f1']:.4f} "
        f"(ep {result['best_epoch']}) | [bold]{' | '.join(summary)}[/bold]"
    )

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def score_tests(cfg: Dict, experiment_id: str, model: torch.nn.Module, device: torch.device, datasets: Dict,
                loaders: Dict, autocast=None, out_root: str = None) -> Tuple[Dict, List[str]]:
    """Val and test predictions of a selected checkpoint with the configured test-time inference (`inference`):
    whole images with dihedral TTA, the `field_cohorts` scored cell by cell when `field` is `cells`, ALL iff
    p(ALL) >= `decision_threshold`. The robustness methods (`wise_alpha`, `stain_norm`, `tent`; src/evaluation/
    robust.py) act on val and on every test cohort alike; TENT adapts a fresh copy per cohort. Writes
    {out_root}/predictions (default: results_dir) and the confusion matrices; returns the test_* result keys and
    one summary line per cohort."""
    inf = cfg["inference"]
    out_root = out_root or abs_path(cfg["results_dir"])
    class_names = cfg["phase1"]["class_names"]
    views = cfg.get("tta_views", 8)
    args = dict(tta_views=views, device=device, autocast=autocast, channels_last=cfg["channels_last"])
    norm = robust.stain_normalizer(inf, abs_path(cfg["randstainna_stats"]), device)
    transform = norm.images if norm is not None else None
    record, val = {}, None
    if inf.get("wise_alpha") is not None:
        model, val, record["wise"] = robust.wise_ft(
            model, cfg["model_name"], inf["wise_alpha"],
            lambda m: predict_images(m, loaders["val"], transform=transform, **args))
    if norm is not None:
        record["stain_norm"] = inf["stain_norm"]
    tent = bool(inf.get("tent", False))
    if tent:
        record["tent"] = {"steps": int(inf["tent_steps"]), "lr": float(inf["tent_lr"]), "batch": robust.TENT_BATCH,
                          "params": "LayerNorm affine", "episodic": "per cohort"}

    def adapted(sampler) -> torch.nn.Module:
        return robust.tent_adapt(model, *sampler, steps=int(inf["tent_steps"]), lr=float(inf["tent_lr"]),
                                 autocast=autocast, channels_last=cfg["channels_last"]) if tent else model

    if tent or val is None:
        val_model = adapted(robust.image_sampler(datasets["val"], device, transform))
        val = predict_images(val_model, loaders["val"], transform=transform, **args)
    if record:
        val["robust"] = record
    _save_predictions(out_root, experiment_id, "val", val)
    result, summary = {}, []
    for key in TEST_SETS:
        cells = inf["field"] == "cells" and key in inf["field_cohorts"]
        inputs = field_inputs(datasets[key], inf["cells"], norm.field if norm is not None else None) if cells else None
        sampler = (robust.crop_sampler(*inputs, crop_scales(inf["cells"])[0], cfg["img_size"], device) if cells
                   else robust.image_sampler(datasets[key], device, transform))
        cohort_model = adapted(sampler)
        test = predict_images(cohort_model, loaders[key], transform=transform, **args)
        test["inference"] = "resize"
        if cells:
            test = predict_fields(cohort_model, datasets[key], test, inf["cells"], img_size=cfg["img_size"],
                                  inputs=inputs, **args)
            test["inference"] = settings_summary(inf["cells"])
        if record:
            test["robust"] = record
            test["inference"] += robust.summary(record)
        p_positive = [row[1] for row in test["probs"]]
        test["threshold"] = resolve_threshold(inf["decision_threshold"], val["y_true"], [r[1] for r in val["probs"]],
                                              p_positive)
        free = threshold_free_metrics(test["y_true"], p_positive)
        dec = decision_metrics(test["y_true"], test["probs"], class_names, threshold=test["threshold"])
        result.update({
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
            f"test_{key}_inference": test["inference"],
        })
        _save_predictions(out_root, experiment_id, f"test_{key}", test)
        save_confusion(
            cm=np.array(dec["confusion_matrix"]),
            class_names=class_names,
            save_path=os.path.join(out_root, "figures", "confusion_matrices", f"{experiment_id}_test_{key}_cm.png"),
            heading=TEST_SET_LABELS.get(key, key),
        )
        flag = " [red](degenerate)[/red]" if dec["degenerate"] else ""
        summary.append(f"{key} AUC {free['roc_auc']:.4f} / F1 {dec['macro_f1']:.4f} (t {dec['threshold']:.3f}){flag}")
    return result, summary


def _save_predictions(out_root: str, experiment_id: str, split: str, predictions: Dict) -> None:
    """Per-image predictions (with file names) of `split` (val, test_<key>) for scripts/evaluate.py: y_pred at the
    file's `threshold` (0.5 when absent); test files also name the inference that produced them."""
    out_dir = os.path.join(out_root, "predictions")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{experiment_id}_{split}.json")
    threshold = predictions.get("threshold", DECISION_THRESHOLD)
    keys = ("y_true", "probs", "names", "threshold", "inference", "cell_probs", "robust")
    payload = {k: v for k, v in predictions.items() if k in keys}
    payload["y_pred"] = [int(row[1] >= threshold) for row in predictions["probs"]]
    # the file marks the run as done for resume, so it appears only complete
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(path + ".tmp", path)


def run_pretraining(augs: List[str], seeds: List[int], epochs: int = None, limit: int = 0,
                    overrides: Dict = None, screen: str = None) -> List[Dict]:
    """`screen` = val-only run tagged phase1_{screen}_seed{S}: no test cohort, no table export."""
    cfg = apply_overrides(load_config(), overrides or {})
    if screen is not None:  # kept apart so no full report ever scores a screening run on a test cohort
        cfg["results_dir"] = f"{cfg['results_dir']}/screen"
    console = get_console()
    p1 = cfg["phase1"]
    epochs = epochs or p1["epochs"]

    if torch.cuda.is_available():
        gpu = f"{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)"
    else:
        gpu = "CPU"

    # Dataset summary computed from disk (no hard-coded counts)
    datasets, _ = build_loaders(cfg, seeds[0], limit, with_tests=screen is None)

    def fmt(k):
        c = class_counts(datasets[k], 2)
        return f"{len(datasets[k]):,} images ({c[0]:,} Normal / {c[1]:,} ALL)"
    train_name = "Train (C-NMC + Bodzas aux cells)" if cfg["aux_train"]["enabled"] else "Train (C-NMC, every fold)"
    cohorts = {train_name: fmt("train"), "Val (LeukemiaAttri crops, other lab)": fmt("val")}
    if screen is None:
        cohorts.update({f"Test: {TEST_SET_LABELS[k]}": fmt(k) for k in TEST_SETS})
    print_header_panel(
        title="MorphoMix: binary training",
        subtitle=("Screening run: val only, no test cohort" if screen is not None else
                  "Train C-NMC, select on LeukemiaAttri; tests: ALL-IDB2 and Aria (each another laboratory)"),
        info_dict={
            "Hardware": f"{gpu} | {cfg['mixed_precision']} mixed precision, batch {cfg['batch_size']}, "
                        f"deterministic {bool(cfg.get('deterministic', False))}",
            "Model": f"{cfg['model_name']} (timm fcmae_ft_in1k weights, new 2-class head)",
            **cohorts,
            "Schedule": f"{epochs} epochs, warmup {p1['warmup_epochs']}, "
                        f"AdamW peak lr {peak_lr(cfg):.1e} (layer decay {p1['layer_decay']}), "
                        f"wd {p1['weight_decay']}, drop path {cfg['drop_path']}",
            "Arms": ", ".join(augs),
            "Seed budget": f"primary {cfg.get('primary_seed', seeds[0])} for every arm; "
                           f"{seeds} for {cfg.get('headline_arms', [])}",
            "Evaluation": f"{cfg.get('tta_views', 8)}-view TTA, ROC-AUC/AUPRC primary, macro-F1 at t = "
                          f"{cfg['inference']['decision_threshold']}; whole fields: {cfg['inference']['field']}",
        }
    )

    plan = [(aug, seed) for aug in augs for seed in seeds_for(cfg, aug, seeds)]
    results = []
    total = len(plan)
    i = 0
    for aug, seed in plan:
        i += 1
        print_ablation_header(step_idx=i, total_steps=total, name=f"{aug} / seed {seed}", desc=experiment_desc(aug))
        results.append(run_one(cfg, aug, seed, epochs, console, limit, tag=screen, with_tests=screen is None))
        if screen is None:
            _export_table(cfg, results)

    print_master_comparison(results)
    console.print("[success]done[/success] phase 1\n")
    return results


def experiment_desc(aug: str) -> str:
    return {
        "basic": "Flips, rotation, colour jitter (no mixing)",
        "hed_jitter": "HED stain jitter (Tellez 2019), cell pixels",
        "randstainna": "RandStainNA virtual Lab template (Shen 2022), cell pixels",
        "stain_mixup": "Macenko stain matrix mixed with an MLL23 one (Chang 2021), cell pixels",
        "morpho_mix": "in-cell MLL23 stain transfer, small cells and background, one cell mask (ours)",
    }.get(aug, aug)


def _export_table(cfg: Dict, results: List[Dict], out_root: str = None) -> None:
    """The phase-1 result table under {out_root}/tables (default: results_dir); the threshold and the inference of
    every cohort are columns of the JSON."""
    metrics = ("roc_auc", "auprc", "macro_f1", "sensitivity_all", "specificity_all", "predicted_positive_rate",
               "degenerate", "threshold", "inference", "macro_f1_secondary", "threshold_secondary",
               "predicted_positive_rate_secondary")
    shown = ("roc_auc", "auprc", "macro_f1", "predicted_positive_rate", "degenerate")
    if cfg["inference"].get("secondary_threshold") is not None:
        shown += ("macro_f1_secondary",)
    rows = [{"augmentation": r["augmentation"], "seed": r["seed"], "best_epoch": r["best_epoch"],
             "val_roc_auc": r.get("roc_auc"), "val_macro_f1": r["macro_f1"],
             **{f"{k}_{m}": r.get(f"test_{k}_{m}") for k in TEST_SETS for m in metrics}} for r in results]
    export_results_table(
        rows, os.path.join(out_root or abs_path(cfg["results_dir"]), TABLE_PREFIX),
        caption=("Trained on C-NMC plus the Bodzas auxiliary cells, selected on LeukemiaAttri val, scored on "
                 "ALL-IDB2 (single cells) and Aria (full fields), each another laboratory; 8-view TTA, ALL "
                 "predicted iff p(ALL) >= the threshold of `inference.decision_threshold` "
                 f"({cfg['inference']['decision_threshold']}); "
                 f"macro_f1_secondary at `secondary_threshold` ({cfg['inference'].get('secondary_threshold')}); "
                 "per-run thresholds and field inference in the JSON. A "
                 "predicted positive rate >= 0.95 or <= 0.05 marks a degenerate run; sensitivity and specificity are "
                 "in the JSON."),
        label="tab:phase1",
        columns=["augmentation", "seed", "val_roc_auc"] + [f"{k}_{m}" for k in TEST_SETS for m in shown],
    )


def parse_args(argv=None):
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aug", nargs="+", default=cfg["augmentations"], choices=cfg["augmentations"])
    p.add_argument("--seed", nargs="+", type=int, default=cfg["seeds"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--limit", type=int, default=0, help="truncate each split to N images (smoke test)")
    p.add_argument("--screen", metavar="TAG", default=None,
                   help="val-only screening run: id phase1_TAG_seedS, test cohorts never evaluated")
    p.add_argument("--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
                   help="config overrides (YAML values); a dotted key such as phase1.base_lr sets a nested value")
    return p.parse_args(argv)


def parse_value(text: str):
    """YAML value, except for two YAML 1.1 quirks: `8e-4` (a float only with a dot) becomes a float too, and
    `no` / `yes` / `on` / `off` stay strings (`mixed_precision=no`); only true / false are booleans."""
    import yaml
    value = yaml.safe_load(text)
    if isinstance(value, bool) and text.strip().lower() not in ("true", "false"):
        return text
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def main(argv=None):
    a = parse_args(argv)
    overrides = {k: parse_value(v) for k, v in (item.split("=", 1) for item in a.set)}
    if a.limit:
        refuse_real_results_dir(apply_overrides(load_config(), overrides)["results_dir"])
    run_pretraining(augs=a.aug, seeds=a.seed, epochs=a.epochs, limit=a.limit, overrides=overrides, screen=a.screen)


if __name__ == "__main__":
    main()
