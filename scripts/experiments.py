"""Experiment runners where a candidate is a config override of one arm: the component ablation (`ablation`), the
val-only screen (`screen`) and the exploratory recipe tuning (`tune`, `tune --proxy`). Every candidate is trained by
`train.run_one` under its own experiment id; the ids, tables and done markers below are read back on resume."""
import argparse
import copy
import json
import os
import re
import time
from typing import Dict, List, Optional

import numpy as np

from src.utils.config import abs_path, apply_overrides, load_config, refuse_real_results_dir, setup_cuda_env
setup_cuda_env()

import torch  # noqa: E402

from scripts import evaluate, train  # noqa: E402
from scripts.train import TEST_SETS  # noqa: E402
from src.utils.exporter import export_results_table  # noqa: E402
from src.utils.logger import boxed_table, get_console, print_ablation_header  # noqa: E402


def candidate_cfg(cfg: Dict, overrides: Dict, results_dir: str) -> Dict:
    """The config of one candidate; apply_overrides rejects an unknown key before anything trains."""
    out = apply_overrides(copy.deepcopy(cfg), overrides)
    out["results_dir"] = results_dir
    return out


def write_json(path: str, payload) -> None:
    """Via a temp file: these files are read back by resume, so they never appear half written."""
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(path + ".tmp", path)


# --- `ablation`: component ablation ---

ABLATION_DOC = """Component ablation."""
ARM_AUG = "morpho_mix"
# (a) has exactly the config of the main morpho_mix run, so it reads that run instead of training a copy
REFERENCE_ARM = "a"
ARMS: Dict[str, Dict] = {
    "a": {"desc": "MorphoMix full (reference = the main morpho_mix run)", "aug": ARM_AUG, "cfg": {}},
    "b": {"desc": "no in-cell stain augmentation (RandStainNA and C1 off)", "aug": ARM_AUG,
          "cfg": {"rsn_prob": 0.0, "appearance_prob": 0.0, "acquisition_prob": 0.0}},  # acquisition belongs to C1
    "c": {"desc": "no small-cell rendering (C2' off)", "aug": ARM_AUG, "cfg": {"small_cell_prob": 0.0}},
    "d": {"desc": "no background randomisation (C3 off)", "aug": ARM_AUG, "cfg": {"use_background": False}},
    "e": {"desc": "RandStainNA (C1) instead of the MLL23 bank transfer", "aug": ARM_AUG,
          "cfg": {"rsn_prob": 0.5, "appearance_prob": 0.0}},
    "f": {"desc": "no auxiliary training cells (C-NMC only)", "aug": ARM_AUG, "cfg": {"aux_train.enabled": False}},
    # controls added after the CBM-style review (2026-10-05): the Bodzas slide-colour cue, and the baselines without
    # the auxiliary cells (arm f is MorphoMix without them)
    "g": {"desc": "auxiliary cells colour-matched to the C-NMC template", "aug": ARM_AUG,
          "cfg": {"aux_train.dir": load_config()["aux_train"]["colour_matched_dir"]}},
    "h": {"desc": "basic, no auxiliary cells", "aug": "basic", "cfg": {"aux_train.enabled": False}},
    "i": {"desc": "hed_jitter, no auxiliary cells", "aug": "hed_jitter", "cfg": {"aux_train.enabled": False}},
    "j": {"desc": "randstainna, no auxiliary cells", "aug": "randstainna", "cfg": {"aux_train.enabled": False}},
    "k": {"desc": "stain_mixup, no auxiliary cells", "aug": "stain_mixup", "cfg": {"aux_train.enabled": False}},
}
ABLATION_TABLE = "tables/component_ablation"  # under results_dir
# the composition was chosen with the test results known (decision log); the table is exploratory
VAL_COLUMNS = ["val_roc_auc", "val_macro_f1"]
TEST_COLUMNS = [f"{k}_{m}" for k in TEST_SETS for m in ("roc_auc", "auprc", "macro_f1")]
TABLE_COLUMNS = VAL_COLUMNS + TEST_COLUMNS


def run_arm(arm: str, seed: int, epochs: Optional[int], limit: int, console) -> Dict:
    spec = ARMS[arm]
    cfg = apply_overrides(load_config(), spec["cfg"])
    # the reference arm reads the main run's artifacts; any arm reuses a finished run whose JSON row was lost
    tag = spec["aug"] if arm == REFERENCE_ARM else f"abl{arm}"
    experiment_id = f"phase1_{tag}_seed{seed}"

    print_ablation_header(0, 0, f"ablation ({arm}) / seed {seed}", spec["desc"])
    p1 = evaluate.phase1_result(cfg, experiment_id)
    if p1 is None:  # not finished: an interrupted run's checkpoint holds no references yet and is retrained
        train.run_one(cfg, spec["aug"], seed, epochs or cfg["phase1"]["epochs"], console, limit=limit, tag=tag)
        p1 = evaluate.phase1_result(cfg, experiment_id)  # val AUC is read back from the checkpoint
    else:
        train.check_references(cfg, spec["aug"], experiment_id)
    row = {"arm": arm, "seed": seed, "description": spec["desc"],
           "val_roc_auc": p1["roc_auc"], "val_macro_f1": p1["macro_f1"]}
    row.update({col: p1[f"test_{col}"] for col in TEST_COLUMNS})
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}


def seed_summary(rows: List[Dict]) -> List[Dict]:
    """Mean and sample sd over seeds per arm; an arm-to-arm gap below the sd is seed noise, not an effect."""
    out = []
    for arm in sorted({r["arm"] for r in rows}):
        runs = [r for r in rows if r["arm"] == arm]
        entry = {"arm": arm, "description": runs[0]["description"], "seeds": len(runs)}
        for col in TABLE_COLUMNS:
            v = np.array([r[col] for r in runs if col in r], dtype=np.float64)
            entry[col] = f"{v.mean():.4f} +/- {v.std(ddof=1):.4f}" if v.size > 1 else (
                round(float(v[0]), 4) if v.size else "n/a")
        out.append(entry)
    return out


def ablation_main(argv=None):
    p = argparse.ArgumentParser(description=ABLATION_DOC, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", nargs="+", default=list(ARMS.keys()), choices=list(ARMS.keys()))
    p.add_argument("--seed", nargs="+", type=int, default=None, help="default: `ablation_seeds` in the config")
    p.add_argument("--epochs", type=int, default=None, help="default: phase1.epochs")
    p.add_argument("--limit", type=int, default=0, help="truncate the splits (smoke test)")
    a = p.parse_args(argv)
    cfg = load_config()
    if a.limit:
        refuse_real_results_dir(cfg["results_dir"])
    seeds = a.seed or cfg["ablation_seeds"]

    console = get_console()
    prefix = abs_path(os.path.join(cfg["results_dir"], ABLATION_TABLE))
    out_json = prefix + "_runs.json"
    rows = []
    if os.path.exists(out_json):
        with open(out_json, encoding="utf-8") as f:
            # keep other cells, redo the requested ones
            rows = [r for r in json.load(f) if not (r["arm"] in a.arms and r["seed"] in seeds)]

    for arm, seed in [(arm, seed) for seed in seeds for arm in a.arms]:
        rows.append(run_arm(arm, seed, a.epochs, a.limit, console))
        rows.sort(key=lambda r: (r["arm"], r["seed"]))
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        write_json(out_json, rows)
        export_results_table(seed_summary(rows), prefix,
                             caption=("Component ablation, mean +/- sd over seeds (per-seed rows in the runs "
                                      "JSON). The composition, the auxiliary cells and the field inference were "
                                      "chosen with the ALL-IDB2 and Aria results known (exploratory; see the "
                                      "unblinding record), and MorphoMix alone was tuned on them (tune --grid final), "
                                      "so the test columns are not evidence of generalisation."),
                             label="tab:ablation",
                             columns=["arm", "description", "seeds"] + TABLE_COLUMNS)
    console.print("[success]done[/success] ablation")


# --- `screen`: val-only screening ---

SCREEN_DOC = """Val-only screening of MorphoMix components and training candidates, with a pre-registered decision rule.

Every candidate differs from its reference by one config change and is trained with `pretrain --screen`, so no
test cohort is ever loaded. Each checkpoint is then scored on the val cohort (LeukemiaAttri crops): clean ROC-AUC,
J over the selection stress families (simulated stain, exposure and scale shift) and the share of CAM energy inside
the central cell.
"""
# Stage 1 (`--lr`, optional), before any component: the shared phase1.base_lr, screened on `basic` so that no
# MorphoMix component shapes the schedule every arm uses. The reference is the a-priori config value (4e-4, the old
# batch-48 head lr unscaled, peak 2e-4), so "nothing eligible" keeps a value that was actually run. The candidates
# are Table 9 of ConvNeXt V2 at batch 128 (2e-4, peak 1e-4), A.3's 1e-4 at batch 32 (8e-4, peak 4e-4; the old
# batch-48 recipe scaled linearly to 128 is base ~1.07e-3) and A.4's 1e-4 at batch 16 (1.6e-3, peak 8e-4).
LR_CANDIDATES: Dict[str, tuple] = {
    "lr4": ("basic", {"phase1.base_lr": 4.0e-4}, None, "reference"),
    "lr2": ("basic", {"phase1.base_lr": 2.0e-4}, "lr4", "lr"),
    "lr8": ("basic", {"phase1.base_lr": 8.0e-4}, "lr4", "lr"),
    "lr16": ("basic", {"phase1.base_lr": 1.6e-3}, "lr4", "lr"),
}
# Stage 2, at the adopted base_lr. name: (arm, config overrides, reference, kind); "add" = new candidate or value,
# "drop" = one component removed
COMPONENT_CANDIDATES: Dict[str, tuple] = {
    "ref": ("morpho_mix", {}, None, "reference"),
    "ema": ("morpho_mix", {"ema_decay": 0.998}, "ref", "add"),
    "acq": ("morpho_mix", {"acquisition_prob": 0.5}, "ref", "add"),
    "noc1": ("morpho_mix", {"rsn_prob": 0.0, "appearance_prob": 0.0, "acquisition_prob": 0.0}, "ref", "drop"),
    "noc2": ("morpho_mix", {"small_cell_prob": 0.0}, "ref", "drop"),  # C2' small cell
    "noc3": ("morpho_mix", {"use_background": False}, "ref", "drop"),
    "c4": ("morpho_mix", {"field_prob": 0.25}, "ref", "add"),
    "basic": ("basic", {}, None, "reference"),
    "basicema": ("basic", {"ema_decay": 0.998}, "basic", "add"),
}
SCREEN_CANDIDATES: Dict[str, tuple] = {**LR_CANDIDATES, **COMPONENT_CANDIDATES}
# a val difference smaller than this is treated as noise
TOL = 0.005
SCREEN_TABLE = "tables/screen_decision"  # under results_dir


def screen_tag(name: str) -> str:
    return f"scr-{name}"


def verdict(kind: str, d_auc: float, d_j: float) -> str:
    """Pre-registered rule; without clear evidence the current config.yaml setting stays."""
    if kind == "lr":  # clean AUC is the checkpoint selection metric; `decide` adopts the best eligible lr
        return "eligible" if (d_auc >= TOL and d_j >= -TOL) else "keep base_lr"
    if kind == "add":
        if d_j >= TOL and d_auc >= -TOL:
            return "adopt"
        return "reject" if (d_j <= -TOL or d_auc <= -TOL) else "inconclusive: keep off"
    if d_j <= -TOL or d_auc <= -TOL:
        return "keep component"
    return "drop component" if (d_j >= TOL and d_auc >= -TOL) else "inconclusive: keep on"


def train_candidates(cfg: Dict, names: List[str], seeds: List[int], epochs: int, limit: int, console) -> None:
    runs_dir = abs_path(f"{cfg['results_dir']}/screen/runs")
    os.makedirs(runs_dir, exist_ok=True)
    for seed in seeds:
        for name in names:
            arm, overrides, _, _ = SCREEN_CANDIDATES[name]
            done = os.path.join(runs_dir, f"phase1_{screen_tag(name)}_seed{seed}.json")
            if os.path.exists(done):  # the checkpoint alone may be from an interrupted run
                screened = candidate_cfg(cfg, overrides, f"{cfg['results_dir']}/screen")
                train.check_references(screened, arm, f"phase1_{screen_tag(name)}_seed{seed}")
                console.print(f"  {screen_tag(name)} seed {seed} [dim]done, skipped[/dim]")
                continue
            result = train.run_pretraining([arm], [seed], epochs, limit=limit, overrides=overrides,
                                           screen=screen_tag(name))
            write_json(done, {"overrides": overrides, "result": result[0]})


def decide(rows: List[Dict], names: List[str], seeds: List[int]) -> List[Dict]:
    by_id = {r["id"]: r for r in rows}

    def metric(name: str, seed: int, key: str) -> float:
        return by_id[f"phase1_{screen_tag(name)}_seed{seed}"][key]

    out = []
    for name in names:
        arm, overrides, ref, kind = SCREEN_CANDIDATES[name]
        if ref is None:
            continue
        # rounded as shown in the table, so the verdict can be recomputed from it
        deltas = {k: round(float(np.mean([metric(name, s, k) - metric(ref, s, k) for s in seeds])), 4)
                  for k in ("clean_auc", "J_selection", "layercam4_in_cell", "evidence_in_cell")}
        out.append({
            "candidate": name, "change": json.dumps(overrides), "vs": ref, "seeds": len(seeds),
            **{f"d_{k}": v for k, v in deltas.items()},
            "verdict": verdict(kind, deltas["clean_auc"], deltas["J_selection"]),
        })
    eligible = [d for d in out if d["verdict"] == "eligible"]
    if eligible:
        best = max(eligible, key=lambda d: d["d_clean_auc"])
        for d in eligible:
            d["verdict"] = "adopt" if d is best else "eligible, not best"
    return out


def adopted_lr(table_json: str) -> Optional[float]:
    """base_lr adopted by the stage-1 table, None when it kept the config value or has not run."""
    if not os.path.exists(table_json):
        return None
    with open(table_json, encoding="utf-8") as f:
        rows = json.load(f)
    return next((json.loads(r["change"])["phase1.base_lr"] for r in rows if r["verdict"] == "adopt"), None)


def screen_main(argv=None) -> int:
    cfg = load_config()
    p = argparse.ArgumentParser(description=SCREEN_DOC, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lr", action="store_true", help="stage 1: screen phase1.base_lr on `basic`, before the components")
    p.add_argument("--only", nargs="+", default=None, metavar="NAME",
                   help=f"candidates to run (default: the whole stage; references are added automatically); with "
                        f"--lr one of {', '.join(LR_CANDIDATES)}, else one of {', '.join(COMPONENT_CANDIDATES)}")
    p.add_argument("--seed", nargs="+", type=int, default=[cfg["primary_seed"]])
    p.add_argument("--epochs", type=int, default=None, help="default: phase1.epochs")
    p.add_argument("--limit", type=int, default=0,
                   help="truncate training to N train (and in-training val) images (smoke test); the screen's val "
                        "scoring always uses the full val cohort")
    a = p.parse_args(argv)
    stage = LR_CANDIDATES if a.lr else COMPONENT_CANDIDATES
    wrong = [n for n in a.only or [] if n not in stage]
    if wrong:
        p.error(f"--only {' '.join(wrong)}: not a {'stage-1 (--lr)' if a.lr else 'stage-2'} candidate; choose from "
                f"{', '.join(stage)}")
    if a.limit:
        refuse_real_results_dir(cfg["results_dir"])
    only = a.only or list(stage)
    names = list(dict.fromkeys([SCREEN_CANDIDATES[n][2] for n in only if SCREEN_CANDIDATES[n][2]] + only))
    table = abs_path(os.path.join(cfg["results_dir"], SCREEN_TABLE + ("_lr" if a.lr else "")))
    lr = adopted_lr(abs_path(os.path.join(cfg["results_dir"], SCREEN_TABLE + "_lr.json")))
    if not a.lr and lr is not None and lr != cfg["phase1"]["base_lr"]:
        raise SystemExit(f"Stage 1 adopted phase1.base_lr = {lr:g} but config.yaml has {cfg['phase1']['base_lr']:g}; "
                         f"set it before screening the components.")
    console = get_console()

    train_candidates(cfg, names, a.seed, a.epochs, a.limit, console)

    ids = {f"phase1_{screen_tag(n)}_seed{s}" for n in names for s in a.seed}
    ckpts = [c for c in evaluate.phase1_checkpoints(cfg, screen=True) if f"phase1_{c[0]}_seed{c[1]}" in ids]
    val = evaluate.load_val_cohort(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = evaluate.screen_on_val(cfg, ckpts, val, int(cfg.get("tta_views", 8)), device, console)

    decisions = decide(rows, names, a.seed)
    rule = (f"the reference is the a-priori config base_lr; a candidate base_lr is eligible when clean val ROC-AUC "
            f"rises by >= {TOL} and J_selection falls by <= {TOL} against it; the eligible one with the largest AUC "
            f"gain is adopted, otherwise the reference base_lr stays." if a.lr else
            f"adopt an addition when J_selection rises by >= {TOL} and clean val ROC-AUC falls by <= {TOL}; keep a "
            f"component when removing it lowers either by >= {TOL}, drop it when removing it raises J_selection by "
            f">= {TOL} without that AUC loss; anything in between leaves config.yaml unchanged.")
    export_results_table(decisions, table,
                         caption=(f"Val-only screening, candidate minus reference averaged over seeds {a.seed}. "
                                  f"Rule fixed before the runs: {rule}"),
                         label="tab:screen_lr" if a.lr else "tab:screen_decision")
    for d in decisions:
        console.print(f"  {d['candidate']:<9} {d['change']:<40} dAUC {d['d_clean_auc']:+.4f} "
                      f"dJ {d['d_J_selection']:+.4f} dCAM {d['d_layercam4_in_cell']:+.3f}  [bold]{d['verdict']}[/bold]")
    return 0


# --- `tune`: exploratory recipe tuning ---

TUNE_DOC = """Exploratory, test-guided tuning of the training recipe (`python main.py tune`).

The test cohorts were unblinded on 2026-10-03 (decision log), so every number here is exploratory and is
reported as such. A candidate is an arm plus config overrides. It is trained by `train.run_one` into
{results_dir}/tuning as phase1_tune-<name>_seed<S>, selected on val like every run and scored on ALL-IDB2 and Aria by
the same call (the configured test-time inference), then scored on the val cohort by `evaluate.screen_row` (selection
stress J and CAM energy in the central cell). Val is the first column of tuning/tables/tune_summary, so a recipe
that helps only the tests stays visible. Nothing under tuning/ is read by `all`, `tables` or `stats`.

`--grid final --adopt` (2026-10-06, the author's decision) tunes the final MorphoMix alone on the test cohorts, on
MorphoMix-only knobs, and writes the candidate the pre-declared rule (`adopt`, ADOPT_MARGIN) picks into the config the
`all` that follows trains; the baselines are not tuned (decision log, change 7).

`--proxy` ranks PROXY_GRID on the laptop first, with config `tune_proxy` on top of every candidate (a fixed 30 %
class- and fold-stratified train subset, 15 epochs, batch 32 x 4 accumulation, its own results_dir); full val and
test cohorts. Proxy numbers only prune the Colab grid. `--proxy --grid proxy-extra` runs PROXY_EXTRA_GRID
(RandStainNA hybrid, LP-FT and MixStyle) into the same results_dir, so one tune_summary holds both batches.
"""
TUNE_DIR = "tuning"  # under results_dir
TUNE_TABLE = "tables/tune_summary"  # under the tuning dir
# Every arm of the first full run peaked on val at epoch 1-3 (inside the 5-epoch warm-up) and then decayed. A 2-4x
# lower lr reaches the same cumulative lr by about epoch 6-10, so 20 epochs leave room for a later peak and still
# show the decay; the reference keeps its early peak, since the first 5 epochs equal those of a 30-epoch run.
TUNE_EPOCHS = 20
# lower lr with stronger layer decay (stem at 0.75^13 = 2 % of the head lr): the backbone barely leaves its
# pretrained features, which the first run forgot
# the config defaults the tune grid ran under; the stored spec holds only the candidate overrides, so these are
# applied first (not stored) and a later change of the defaults leaves every candidate as it was trained
TUNE_DEFAULTS = {"appearance_prob": 0.5, "rsn_prob": 0.0, "aux_train.enabled": False}
BASE = {"phase1.base_lr": 1.0e-4, "phase1.layer_decay": 0.75}
# chosen before any tuning run: BASE plus every recipe-level addition (EMA, label smoothing, low resolution)
BEST = {**BASE, "ema_decay": 0.999, "phase1.label_smoothing": 0.1, "lowres_prob": 0.5}
BEST_NOEMA = {**BASE, "phase1.label_smoothing": 0.1, "lowres_prob": 0.5}
# MorphoMix composition, pinned in every morpho_mix candidate (on top of TUNE_DEFAULTS). C2 (cell zoom) was removed
# after the first cross-lab ablation; C2' (small cell) and C4 (multi-cell field) replace it (test-guided,
# decision log). NOC2 is C1 + C3 alone (no cell-size component; the name is kept from the first grid)
NOC2 = {"small_cell_prob": 0.0, "field_prob": 0.0}
C2P_C4 = {"small_cell_prob": 0.25, "field_prob": 0.25}
# LP-FT (Kumar et al., ICLR 2022): 2 epochs of a linear probe (whole backbone frozen) at a constant head lr of 1e-3,
# 5-20x the schedule's head lr, so the probe gets near a fitted head (50 steps in the proxy, 166 at full scale);
# then the usual warm-up + cosine over the remaining epochs
LPFT = {"phase1.freeze_stages": 4, "phase1.freeze_epochs": 2, "phase1.probe_lr": 1.0e-3}
# MixStyle (Zhou et al., ICLR 2021) with the paper's p 0.5 and alpha 0.1 after the first two stages
MIXSTYLE = {"mixstyle_p": 0.5, "mixstyle_alpha": 0.1, "mixstyle_stages": [0, 1]}
# RandStainNA hybrid (rsn_prob 0.5) on the composition of ref-c2p: in proxy batch 1 randstainna was best on the tests
# and ref-c2p best on val
HYB_C2P = {**NOC2, "small_cell_prob": 0.25, "rsn_prob": 0.5}
# name: (arm, config overrides, default seeds). The "ref-" candidates keep the first run's recipe (base_lr 4e-4,
# layer decay 0.9); each base "+" candidate adds one change to `base`, so its effect reads against it
TUNE_CANDIDATES: Dict[str, tuple] = {
    "ref-noc2": ("morpho_mix", NOC2, (42,)),
    "ref-c2p": ("morpho_mix", {**NOC2, "small_cell_prob": 0.25}, (42,)),
    "ref-c4": ("morpho_mix", {**NOC2, "field_prob": 0.25}, (42,)),
    "ref-c2p-c4": ("morpho_mix", {**NOC2, **C2P_C4}, (42,)),
    "lr1-noc2": ("morpho_mix", {"phase1.base_lr": 1.0e-4, **NOC2}, (42,)),
    "base": ("morpho_mix", {**BASE, **NOC2}, (42,)),
    "ema-noc2": ("morpho_mix", {**BASE, **NOC2, "ema_decay": 0.999}, (42,)),  # ~1000-step horizon
    "ls-noc2": ("morpho_mix", {**BASE, **NOC2, "phase1.label_smoothing": 0.1}, (42,)),  # train loss 0.03, val 0.8-2.4
    "lowres-noc2": ("morpho_mix", {**BASE, **NOC2, "lowres_prob": 0.5}, (42,)),  # whole-image lowres vs C2'
    "base-c2p": ("morpho_mix", {**BASE, **NOC2, "small_cell_prob": 0.25}, (42,)),
    "base-c4": ("morpho_mix", {**BASE, **NOC2, "field_prob": 0.25}, (42,)),
    "base-c2p-c4": ("morpho_mix", {**BASE, **NOC2, **C2P_C4}, (42,)),
    "best-noc2": ("morpho_mix", {**BEST, **NOC2}, (42, 43, 44)),
    "best-c2p-c4": ("morpho_mix", {**BEST, **NOC2, **C2P_C4}, (42, 43, 44)),
    "best-basic": ("basic", BEST, (42,)),
    "ref-rsn": ("randstainna", {}, (42,)),  # the baseline at the recipe of the ref- candidates
    "best-rsn": ("randstainna", BEST, (42, 43, 44)),  # the best baseline on ALL-IDB2 in the first run
    # training-time robustness methods (ROBUST_KNOBS), run by name (--only) or by `--proxy --grid proxy-extra`
    "lpft-ref-noc2": ("morpho_mix", {**NOC2, **LPFT}, (42, 43)),
    "lpft-best-noc2": ("morpho_mix", {**BEST, **NOC2, **LPFT}, (42, 43)),
    "mixstyle-ref-noc2": ("morpho_mix", {**NOC2, **MIXSTYLE}, (42, 43)),
    "mixstyle-best-noc2": ("morpho_mix", {**BEST, **NOC2, **MIXSTYLE}, (42, 43)),
    "ref-c2p-c4-lpft": ("morpho_mix", {**NOC2, **C2P_C4, **LPFT}, (42, 43)),
    "ref-c2p-c4-mixstyle": ("morpho_mix", {**NOC2, **C2P_C4, **MIXSTYLE}, (42, 43)),
    "best-c2p-c4-lpft": ("morpho_mix", {**BEST, **NOC2, **C2P_C4, **LPFT}, (42, 43)),
    "best-c2p-c4-mixstyle": ("morpho_mix", {**BEST, **NOC2, **C2P_C4, **MIXSTYLE}, (42, 43)),
    # BEST without EMA: in the proxy (~375 optimizer steps) EMA 0.999 still holds ~69 % of the start weights
    "bestne-c2p-c4": ("morpho_mix", {**BEST_NOEMA, **NOC2, **C2P_C4}, (42, 43)),
    "bestne-c2p-c4-mixstyle": ("morpho_mix", {**BEST_NOEMA, **NOC2, **C2P_C4, **MIXSTYLE}, (42, 43)),
    "rsn-mixstyle": ("randstainna", MIXSTYLE, (42, 43)),  # MixStyle on the baseline too, at the recipe of ref-rsn
    # hyb-c2p: RandStainNA p 0.5, C1 (appearance_prob 0.5 of TUNE_DEFAULTS) on the rest (0.25 overall); rsnonly:
    # no C1, so it reads as C2' + C3 on top of the randstainna arm
    "hyb-c2p": ("morpho_mix", HYB_C2P, (42, 43)),
    "hyb-c2p-rsnonly": ("morpho_mix", {**HYB_C2P, "appearance_prob": 0.0, "acquisition_prob": 0.0}, (42, 43)),
    "hyb-c2p-rsnonly-bodzas": ("morpho_mix", {**HYB_C2P, "appearance_prob": 0.0, "acquisition_prob": 0.0,
                                              "aux_train.enabled": True}, (42, 43)),  # + Bodzas lymphoblasts
}
# a candidate that sets one of these is a robustness candidate, outside the Colab GRID
ROBUST_KNOBS = ("phase1.probe_lr", "mixstyle_p")
# The Colab grid (default, ~2.1 A100 h): the ref-, hyb- and robustness candidates are left out. Dropped from the
# earlier grid: ref, lr2 (between ref and lr1), frz2 (hard freeze, the soft layer-decay freeze is in base) and strong
# (C1 / C3 strength)
GRID = tuple(n for n, (_, o, _) in TUNE_CANDIDATES.items()
             if not n.startswith(("ref-", "hyb-")) and not o.keys() & ROBUST_KNOBS)
# `--proxy` (config `tune_proxy`): the laptop ranking, every candidate with PROXY_SEEDS
PROXY_GRID = ("ref-noc2", "ref-c2p", "ref-c4", "ref-c2p-c4", "best-noc2", "best-c2p-c4", "ref-rsn", "best-rsn")
# `--proxy --grid proxy-extra`, the second laptop batch, into the same results_dir: the RandStainNA hybrids first, then
# LP-FT and MixStyle on the composition of ref-c2p-c4 / best-c2p-c4, on ref-noc2 (the best proxy val of batch 1) and on
# ref-rsn. The best-noc2 ones are left out: best-noc2 collapsed in batch 1 (EMA 0.999 over the proxy's ~375 optimizer
# steps keeps ~70 % of the step-1 weights)
PROXY_EXTRA_GRID = ("hyb-c2p", "hyb-c2p-rsnonly", "ref-c2p-c4-lpft", "ref-c2p-c4-mixstyle", "rsn-mixstyle",
                    "lpft-ref-noc2", "mixstyle-ref-noc2", "bestne-c2p-c4", "bestne-c2p-c4-mixstyle")
PROXY_GRIDS = {"proxy": PROXY_GRID, "proxy-extra": PROXY_EXTRA_GRID}
# 2026-10-06, the author's test-guided tuning of the final MorphoMix (C1 MLL23 + C2' + C3, the former arm e), on
# Colab at the final recipe (30 epochs, batch 128) before the final run; only MorphoMix is tuned, the baselines are
# not, so its test results are not evidence of generalisation (decision log). Pinned in every candidate:
FINAL_E = {"appearance_prob": 0.5, "rsn_prob": 0.0, "acquisition_prob": 0.0, "aux_train.enabled": True}
FINAL_CANDIDATES = {
    "e-ref": {},
    "e-noc3": {"use_background": False},
    "e-noc2": {"small_cell_prob": 0.0},
    "e-app08": {"appearance_prob": 0.8},
    "e-c3low": {"background_prob": 0.15},
}  # MorphoMix-only knobs: a shared key (the lr) would retrain the baselines at a value picked on MorphoMix's tests
# seeds outside `seeds`, so no reported run is one the choice was made on
FINAL_TUNE_SEEDS = (101, 102)
TUNE_CANDIDATES.update({n: ("morpho_mix", {**FINAL_E, **o}, FINAL_TUNE_SEEDS) for n, o in FINAL_CANDIDATES.items()})
COLAB_GRIDS = {"final": tuple(FINAL_CANDIDATES)}
# --adopt, written down before the final grid ran: the candidate with the highest seed-mean of the ALL-IDB2 and Aria
# ROC-AUC average; it replaces e-ref only when it beats e-ref by at least ADOPT_MARGIN (seed noise is ~0.02)
ADOPT_MARGIN = 0.01
PROXY_SEEDS = (42, 43)
TEST_METRICS = {"roc_auc": "auc", "auprc": "auprc", "macro_f1": "f1", "predicted_positive_rate": "pos"}


def tune_tag(name: str) -> str:
    return f"tune-{name}"


def change(overrides: Dict) -> str:
    return " ".join(f"{k.split('.')[-1]}={v:g}" if isinstance(v, float) else f"{k.split('.')[-1]}={v}"
                    for k, v in overrides.items()) or "current recipe"


def summary_row(name: str, seed: int, done: Dict) -> Dict:
    r, v, spec = done["result"], done["val_screen"], done["spec"]
    curve = [c["val_roc_auc"] for c in r["curve"]]
    row = {"candidate": name, "arm": spec["arm"], "seed": seed, "val_auc": r["roc_auc"], "best_ep": r["best_epoch"],
           "last_auc": curve[-1], "near_best": sum(x >= r["roc_auc"] - 0.01 for x in curve),
           "J_sel": v["J_selection"], "cam_lc4": v["layercam4_in_cell"], "cam_ev": v["evidence_in_cell"]}
    row.update({f"{k}_{s}": r[f"test_{k}_{m}"] for k in TEST_SETS for m, s in TEST_METRICS.items()})
    row.update({"train_min": round(done["seconds"] / 60, 1), "epochs": len(curve), "change": change(spec["overrides"]),
                "last_train_loss": r["curve"][-1]["train_loss"], "last_val_loss": r["curve"][-1]["val_loss"]})
    return {k: round(x, 4) if isinstance(x, float) else x for k, x in row.items()}


def export_summary(tune_dir: str, console) -> Optional[List[Dict]]:
    """tune_summary.{md,tex,json} from every finished run under tune_dir/runs, in candidate order."""
    runs_dir = abs_path(f"{tune_dir}/runs")
    rows = []
    for name in TUNE_CANDIDATES:
        pattern = re.compile(rf"phase1_{re.escape(tune_tag(name))}_seed(\d+)\.json")
        for seed in sorted(int(m[1]) for m in map(pattern.fullmatch, os.listdir(runs_dir)) if m):
            with open(os.path.join(runs_dir, f"phase1_{tune_tag(name)}_seed{seed}.json"), encoding="utf-8") as f:
                rows.append(summary_row(name, seed, json.load(f)))
    if not rows:
        return None
    names = dict.fromkeys(r["candidate"] for r in rows)
    legend = "; ".join(f"{n}: {TUNE_CANDIDATES[n][0]}, {change(TUNE_CANDIDATES[n][1])}" for n in names)
    export_results_table(rows, abs_path(f"{tune_dir}/{TUNE_TABLE}"), label="tab:tune", caption=(
        "Exploratory, test-guided recipe tuning (test cohorts unblinded 2026-10-03; not evidence of generalisation; "
        "the final grid tunes MorphoMix alone on the test cohorts). "
        "Checkpoints selected on LeukemiaAttri val ROC-AUC (val_auc, at best_ep; last_auc at the last epoch; near_best "
        "= epochs within 0.01 of the best). J_sel and CAM energy in the central cell (cam_lc4: Layer-CAM stage 4, "
        "cam_ev: evidence map) on val; the cell mask is the one MorphoMix trains with, so CAM scores are partly "
        "circular for morpho_mix. Tests: ROC-AUC, AUPRC, macro-F1 and predicted positive rate at the run threshold. "
        f"train_min: training and test scoring. Candidates: {legend}."),
        columns=["candidate", "seed", "val_auc", "best_ep", "last_auc", "near_best", "J_sel", "cam_lc4", "cam_ev"]
        + [f"{k}_{s}" for k in TEST_SETS for s in TEST_METRICS.values()] + ["train_min"])
    table = boxed_table("Tuning")
    for col in ("candidate", "seed", "val AUC", "ep", "ALL-IDB2 AUC", "Aria AUC", "CAM in cell"):
        table.add_column(col, justify="left" if col == "candidate" else "right")
    for r in rows:
        table.add_row(r["candidate"], str(r["seed"]), f"{r['val_auc']:.4f}", str(r["best_ep"]),
                      f"{r['allidb2_auc']:.4f}", f"{r['aria_auc']:.4f}", f"{r['cam_lc4']:.3f}")
    console.print(table)
    console.print(f"ok {len(rows)} runs -> {tune_dir}/{TUNE_TABLE}.md")
    return rows


def tune_main(argv=None) -> int:
    from scripts.run_all import check_fingerprint  # run_all imports this module
    cfg = load_config()
    p = argparse.ArgumentParser(description=TUNE_DOC, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", default=None, choices=list(TUNE_CANDIDATES), metavar="NAME",
                   help=f"candidates to run (default: the Colab grid, {', '.join(GRID)}; with --proxy "
                        f"{', '.join(PROXY_GRID)}); any of {', '.join(TUNE_CANDIDATES)}")
    p.add_argument("--proxy", action="store_true",
                   help=f"local proxy ranking: config `tune_proxy` on top of every candidate, seeds {PROXY_SEEDS}")
    p.add_argument("--grid", choices=list(PROXY_GRIDS) + list(COLAB_GRIDS), default=None,
                   help=f"with --proxy, the candidates to run (default: proxy); proxy-extra: "
                        f"{', '.join(PROXY_EXTRA_GRID)}; without --proxy, final: {', '.join(COLAB_GRIDS['final'])}")
    p.add_argument("--adopt", action="store_true",
                   help="after the final grid, write the candidate chosen by the pre-declared rule into the config")
    p.add_argument("--seed", nargs="+", type=int, default=None, help="default: each candidate's own seeds")
    p.add_argument("--epochs", type=int, default=None, help=f"default: {TUNE_EPOCHS} (tune_proxy.epochs with --proxy)")
    p.add_argument("--limit", type=int, default=0, help="truncate every split, val scoring included (smoke test)")
    p.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    a = p.parse_args(argv)
    if a.grid in PROXY_GRIDS and not a.proxy or a.grid in COLAB_GRIDS and a.proxy:
        p.error(f"--grid {a.grid} needs {'--proxy' if a.grid in PROXY_GRIDS else 'no --proxy'}")
    if a.adopt and (a.grid != "final" or a.seed or "e-ref" not in (a.only or FINAL_CANDIDATES)
                    or not set(a.only or FINAL_CANDIDATES) <= set(FINAL_CANDIDATES)):
        p.error("--adopt needs --grid final, its own seeds and e-ref among the final candidates only")
    proxy = dict(cfg["tune_proxy"]) if a.proxy else None
    epochs = a.epochs or (proxy["epochs"] if proxy else TUNE_EPOCHS)
    if proxy:
        del proxy["epochs"]
        cfg = apply_overrides(cfg, proxy)
    if a.limit:
        refuse_real_results_dir(cfg["results_dir"])
    tune_dir = f"{cfg['results_dir']}/{TUNE_DIR}"
    names = a.only or (PROXY_GRIDS[a.grid or "proxy"] if proxy else COLAB_GRIDS[a.grid] if a.grid else GRID)
    seeds = {name: a.seed or (PROXY_SEEDS if proxy else TUNE_CANDIDATES[name][2]) for name in names}
    plan = [(name, seed) for name in names for seed in seeds[name]]
    cfgs = {name: candidate_cfg(cfg, {**TUNE_DEFAULTS, **TUNE_CANDIDATES[name][1]}, tune_dir) for name, _ in plan}
    console = get_console()
    runs_dir = abs_path(f"{tune_dir}/runs")
    if not a.dry_run:
        os.makedirs(runs_dir, exist_ok=True)
        check_fingerprint({**cfg, "results_dir": tune_dir}, console)

    val, cells = None, None
    views = int(cfg.get("tta_views", 8))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for i, (name, seed) in enumerate(plan, start=1):
        arm, overrides, _ = TUNE_CANDIDATES[name]
        tag = tune_tag(name)
        eid = f"phase1_{tag}_seed{seed}"
        marker = os.path.join(runs_dir, f"{eid}.json")
        # JSON round trip, so it compares equal to the stored copy
        spec = {"arm": arm, "overrides": overrides, "epochs": epochs, "limit": a.limit}
        if proxy:
            spec["proxy"] = proxy
        spec = json.loads(json.dumps(spec))
        if os.path.exists(marker):  # the checkpoint alone may be from an interrupted run
            with open(marker, encoding="utf-8") as f:
                stored = json.load(f)["spec"]
            if stored != spec:
                raise SystemExit(f"{eid} was trained as {stored}, the candidate is now {spec}; give the changed "
                                 f"candidate a new name or delete {marker}")
            train.check_references(cfgs[name], arm, eid)
            console.print(f"  {eid:<42}[dim]done, skipped[/dim]")
            continue
        if a.dry_run:
            console.print(f"  {eid:<42}[dim]{arm}, {change(overrides)}, {epochs} epochs[/dim]")
            continue
        print_ablation_header(i, len(plan), f"{tag} / seed {seed}", f"{arm}: {change(overrides)}")
        t0 = time.time()
        result = train.run_one(cfgs[name], arm, seed, epochs, console, a.limit, tag=tag)
        seconds = time.time() - t0
        if val is None:  # loaded once, and only when something was trained
            val = evaluate.load_val_cohort(cfg, a.limit)
            cells = evaluate.val_cell_masks(cfg, val)
        row = evaluate.screen_row(cfgs[name], tag, seed, result["checkpoint"], val, cells, views, device, console)
        write_json(marker, {"spec": spec, "result": result, "val_screen": row, "seconds": seconds})

    if not a.dry_run:
        rows = export_summary(tune_dir, console)
        if a.adopt:
            adopt(rows, names, tune_dir, console)
    return 0


def adopt(rows: List[Dict], names, tune_dir: str, console) -> None:
    """The --adopt rule (ADOPT_MARGIN) over the final candidates; writes tune_dir/adopted.json and the chosen
    overrides into the config `load_config` reads, so the `all` that follows trains the chosen MorphoMix."""
    score = {}
    for name in names:
        runs = [r for r in rows if r["candidate"] == name]
        if runs:
            score[name] = float(np.mean([(r["allidb2_auc"] + r["aria_auc"]) / 2 for r in runs]))
    best = max(score, key=score.get)
    chosen = best if score[best] >= score["e-ref"] + ADOPT_MARGIN else "e-ref"
    overrides = FINAL_CANDIDATES[chosen]
    write_json(abs_path(f"{tune_dir}/adopted.json"), {"rule": f"max seed-mean (ALL-IDB2 + Aria ROC-AUC) / 2, "
                                                      f"over e-ref by >= {ADOPT_MARGIN}", "scores": score,
                                                      "chosen": chosen, "overrides": overrides})
    config_file = os.environ.get("MORPHOMIX_CONFIG", "configs/config.yaml")  # the file load_config reads
    set_config_values(abs_path(config_file), overrides)
    scores = ", ".join(f"{k} {v:.4f}" for k, v in score.items())
    console.print(f"adopted {chosen} ({change(overrides)}); scores {scores}")


def set_config_values(path: str, overrides: Dict) -> None:
    """Rewrites the value of each (dotted, at most one level deep) key in place, keeping comments and layout."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    for key, value in overrides.items():
        section, _, leaf = key.rpartition(".")
        indent, start = ("  ", lines.index(f"{section}:") + 1) if section else ("", 0)
        pattern = re.compile(rf"^{indent}{re.escape(leaf)}:(\s*)\S+(.*)$")
        i = next(i for i in range(start, len(lines)) if pattern.match(lines[i]))
        text = str(value).lower() if isinstance(value, bool) else str(value)
        lines[i] = pattern.sub(lambda m: f"{indent}{leaf}:{m[1]}{text}{m[2]}", lines[i])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
