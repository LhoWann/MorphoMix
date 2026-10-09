"""Audits of the trained models and of the cohorts; every analysis is exploratory and post hoc (the test cohorts
were unblinded during development, docs/decision_log.md).

- `python main.py shortcut`: background-only reference, residual AUC, background swap and removal (section 1).
- `python main.py leakaudit`: exact and perceptual duplicates between cohorts (section 2).
- `python main.py fieldaudit`: the whole-field (Aria) detector and field scoring (section 3).
- `python main.py confirm`: the pre-specified replicate LeukemiaAttri L_100X_C1 (section 4).

Each writes to results_dir/tables; the section docstrings below give the details."""
import argparse
import glob
import hashlib
import itertools
import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

from src.utils.config import abs_path, load_config, setup_cuda_env
setup_cuda_env()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.fft import dctn  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from scripts import foundation, train  # noqa: E402
from scripts.evaluate import analysis_groups, cohort_metrics, load_phase1_model, method_level_tests  # noqa: E402
from src.augmentations.transforms import IMAGENET_MEAN, IMAGENET_STD, get_val_transforms  # noqa: E402
from src.cam.blast_prior import extract_blast_cell_prior  # noqa: E402
from src.cam.metrics import central_cell_mask  # noqa: E402
from src.datasets.dataset import BinaryLeukemiaDataset  # noqa: E402
from src.datasets.leukemiaattri import build_val_cohort  # noqa: E402
from src.datasets.loaders import eval_loader  # noqa: E402
from src.evaluation.field_inference import (  # noqa: E402
    crop_cells, detect_cells, field_scores, score_crops, white_cell_mask
)
from src.evaluation.predict import autocast_for, predict_images  # noqa: E402
from src.evaluation.tta import tta_logits  # noqa: E402
from src.utils.exporter import export_results_table  # noqa: E402
from src.utils.logger import get_console  # noqa: E402
from src.utils.seed import setup_run  # noqa: E402

# --------------------------------------------------------------------------------------------------------------------
# 1. Shortcut analyses
# --------------------------------------------------------------------------------------------------------------------
# Shortcut analyses of every phase-1 checkpoint (`python main.py shortcut`); exploratory, the tests are unblinded.
#
# 1. Background-only reference: per image, features of the pixels outside the cell (single cells: the central Azure-B
#    cell, `central_cell_mask`; Aria fields: outside `white_cell_mask`): Lab mean and sd, HSV saturation and Laplacian
#    variance there (`BACKGROUND`), plus the Laplacian variance of the whole image, which also sees the cell. A
#    label-trained logistic regression on `BACKGROUND` alone (5-fold CV, grouped where groups exist) measures how much
#    label information the background carries in that cohort (an upper bound, not a comparator); the same with every
#    feature is written as the image-statistics reference.
# 2. Residual AUC: logit p(ALL) of each run regressed on `BACKGROUND` by least squares (no labels); the ROC-AUC of the
#    residual bounds the discrimination left after the linear background dependence is removed (cell and background
#    share the slide's stain, so the regression also removes some cell signal).
# 3. Background swap on the single-cell cohorts (ALL-IDB2, val), 1 view: every image keeps its central cell (mask
#    dilated 3 px) on the inpainted, cell-free background of a random image of the other class (`other`) or, as the
#    control, of the same class (`same`), averaged over `DONORS` donor draws; `none` puts the cell on black (the C-NMC
#    format, background removed at test time). A class-specific background effect is auc_same - auc_other.
# Writes results_dir/tables/shortcut_*.

COHORTS = {"allidb2": "test_allidb2", "aria": "test_aria", "val": "val"}
SWAP_COHORTS = ("allidb2", "val")
GROUPS = {"allidb2": "data/processed/splits/allidb2_groups.csv", "aria": "data/processed/splits/aria_groups.csv"}
FEATURES = ["bg_L", "bg_a", "bg_b", "bg_Lsd", "bg_asd", "bg_bsd", "bg_sat", "bg_lapvar", "lapvar"]
BACKGROUND = FEATURES[:-1]  # the whole-image Laplacian variance also sees the cell
DONORS = 3
CKPT = re.compile(r"phase1_(?P<arm>.+)_seed(?P<seed>\d+)_best\.pt$")


def cohort_images(cfg: Dict, key: str):
    root = abs_path(cfg["phase1"]["data_dir"])
    names, ys = [], []
    for y, cls in enumerate(cfg["phase1"]["class_names"]):
        files = sorted(glob.glob(os.path.join(root, COHORTS[key], cls, "*.png")))
        names += files
        ys += [y] * len(files)
    return names, np.asarray(ys)


def background_features(rgb: np.ndarray, field: bool, tau: float) -> List[float]:
    cell = (white_cell_mask(rgb) > 0) if field else (central_cell_mask(rgb, tau) > 0)
    bg = ~cell if (~cell).sum() > 100 else np.ones_like(cell)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    sat = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1].astype(np.float32)
    lap = cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.CV_32F)
    return [*lab[bg].mean(0), *lab[bg].std(0), sat[bg].mean(), lap[bg].var(), lap.var()]


def feature_table(cfg: Dict, key: str, cache_dir: str) -> pd.DataFrame:
    cache = os.path.join(cache_dir, f"shortcut_features_{key}.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    files, ys = cohort_images(cfg, key)
    tau = float(cfg["morpho_threshold"])
    rows = [{"name": os.path.basename(p), "y": int(y),
             **dict(zip(FEATURES, background_features(np.asarray(Image.open(p).convert("RGB")), key == "aria", tau)))}
            for p, y in zip(files, ys)]
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


def groups_for(cfg: Dict, key: str, names) -> Optional[np.ndarray]:
    if key == "val":  # crops of one slide share its background
        return analysis_groups(cfg, "val", list(names))
    if key not in GROUPS:
        return None
    g = pd.read_csv(abs_path(GROUPS[key]))
    m = dict(zip(g["file"], g["class_name"] + "_" + g["group"].astype(str)))  # group ids are per class
    return np.array([m.get(n, f"solo_{n}") for n in names])


def cv_auc(x: np.ndarray, y: np.ndarray, g: Optional[np.ndarray]) -> float:
    folds = (GroupKFold(5).split(x, y, g) if g is not None
             else StratifiedKFold(5, shuffle=True, random_state=0).split(x, y))
    oof = np.zeros(len(y))
    for tr, te in folds:
        oof[te] = LogisticRegression(max_iter=2000).fit(x[tr], y[tr]).predict_proba(x[te])[:, 1]
    return round(float(roc_auc_score(y, oof)), 4)


def residual_analysis(cfg: Dict, tables: str, console) -> None:
    rows, reference, image_reference = [], {}, {}
    for key in COHORTS:
        f = feature_table(cfg, key, tables)
        y = f["y"].values
        g = groups_for(cfg, key, f["name"])
        x = StandardScaler().fit_transform(f[BACKGROUND].values)
        reference[key] = cv_auc(x, y, g)
        image_reference[key] = cv_auc(StandardScaler().fit_transform(f[FEATURES].values), y, g)
        index = {n: i for i, n in enumerate(f["name"])}
        suffix = "val" if key == "val" else f"test_{key}"
        for path in sorted(glob.glob(abs_path(os.path.join(cfg["results_dir"], "predictions",
                                                           f"phase1_*_seed*_{suffix}.json")))):
            arm, seed = re.match(r"phase1_(.+)_seed(\d+)_", os.path.basename(path)).groups()
            with open(path, encoding="utf-8") as fh:
                pred = json.load(fh)
            order = [index[os.path.basename(n)] for n in pred["names"]]
            yy = np.asarray(pred["y_true"])
            if not np.array_equal(yy, y[order]):
                raise RuntimeError(f"label mismatch between {path} and the {key} images")
            p = np.clip(np.asarray(pred["probs"])[:, 1], 1e-6, 1 - 1e-6)
            s = np.log(p / (1 - p))
            design = np.c_[np.ones(len(order)), x[order]]
            resid = s - design @ np.linalg.lstsq(design, s, rcond=None)[0]
            rows.append({"cohort": key, "arm": arm, "seed": int(seed), "auc": roc_auc_score(yy, s),
                         "auc_residual": roc_auc_score(yy, resid), "r2_on_background": 1 - resid.var() / s.var()})
    pd.DataFrame(rows).to_csv(os.path.join(tables, "shortcut_residual_runs.csv"), index=False)
    with open(os.path.join(tables, "shortcut_background_only_auc.json"), "w", encoding="utf-8") as fh:
        json.dump(reference, fh, indent=2)
    with open(os.path.join(tables, "shortcut_image_stats_auc.json"), "w", encoding="utf-8") as fh:
        json.dump(image_reference, fh, indent=2)
    for name, ref in (("background-only", reference), ("image-statistics", image_reference)):
        console.print(f"{name} ROC-AUC: " + ", ".join(f"{k} {v:.3f}" for k, v in ref.items()))


def swap_inputs(cfg: Dict, key: str) -> Dict[str, List[np.ndarray]]:
    """Original images and every swap condition; donors drawn with a fixed seed per draw."""
    files, ys = cohort_images(cfg, key)
    imgs = [np.asarray(Image.open(p).convert("RGB")) for p in files]
    tau, kernel = float(cfg["morpho_threshold"]), np.ones((7, 7), np.uint8)
    masks = [cv2.dilate((central_cell_mask(im, tau) > 0).astype(np.uint8), kernel) for im in imgs]
    empty = [cv2.inpaint(im, cv2.dilate(m, kernel), 9, cv2.INPAINT_TELEA) for im, m in zip(imgs, masks)]
    out = {"original": imgs, "none": [im * m[..., None] for im, m in zip(imgs, masks)]}
    for cond in ("other", "same"):
        for d in range(DONORS):
            rng = np.random.default_rng(d)
            pool = {c: np.flatnonzero((ys != c) if cond == "other" else (ys == c)) for c in (0, 1)}
            donors = [rng.choice(pool[y][pool[y] != i]) for i, y in enumerate(ys)]
            out[f"{cond}{d}"] = [np.where(m[..., None] > 0, im, empty[j]) for im, m, j in zip(imgs, masks, donors)]
    return {"y": ys, **out}


@torch.no_grad()
def p_all(model, imgs: List[np.ndarray], device: torch.device, tf) -> np.ndarray:
    out = []
    for i in range(0, len(imgs), 128):
        x = torch.stack([tf(Image.fromarray(a)) for a in imgs[i:i + 128]]).to(device)
        out.append(model(x).float().softmax(1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def swap_analysis(cfg: Dict, tables: str, console) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = get_val_transforms(cfg["img_size"])
    data = {key: swap_inputs(cfg, key) for key in SWAP_COHORTS}
    for key, d in data.items():  # a strip of original / other / same / none for the figure and the reader
        picks = np.linspace(0, len(d["y"]) - 1, 8).astype(int)
        strip = np.concatenate([np.concatenate([d[c][j] for c in ("original", "other0", "same0", "none")], 0)
                                for j in picks], 1)
        Image.fromarray(strip).save(os.path.join(tables, f"shortcut_swap_examples_{key}.png"))
    rows = []
    ckpts = sorted(glob.glob(abs_path(os.path.join(cfg["results_dir"], "checkpoints", "phase1_*_seed*_best.pt"))))
    for path in ckpts:
        m = CKPT.search(os.path.basename(path))
        model = load_phase1_model(cfg, path, device)
        for key, d in data.items():
            y, p0 = d["y"], p_all(model, d["original"], device, tf)
            row = {"cohort": key, "arm": m["arm"], "seed": int(m["seed"]), "auc": roc_auc_score(y, p0),
                   "auc_none": roc_auc_score(y, p_all(model, d["none"], device, tf))}
            for cond in ("other", "same"):
                ps = [p_all(model, d[f"{cond}{k}"], device, tf) for k in range(DONORS)]
                row[f"auc_{cond}"] = float(np.mean([roc_auc_score(y, p) for p in ps]))
                row[f"dp_towards_other_{cond}"] = float(np.mean([np.mean(np.where(y == 1, p0 - p, p - p0))
                                                                 for p in ps]))
            row["auc_drop_other"] = row["auc"] - row["auc_other"]
            row["auc_drop_same"] = row["auc"] - row["auc_same"]
            row["class_specific_effect"] = row["auc_same"] - row["auc_other"]
            rows.append(row)
        console.print(f"  {os.path.basename(path)}")
        del model
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(os.path.join(tables, "shortcut_bg_swap_runs.csv"), index=False)


def shortcut_main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-swap", action="store_true", help="residual analysis only (CPU, no checkpoints needed)")
    a = p.parse_args(argv)
    cfg = load_config()
    console = get_console()
    tables = abs_path(os.path.join(cfg["results_dir"], "tables"))
    os.makedirs(tables, exist_ok=True)
    residual_analysis(cfg, tables, console)
    if not a.skip_swap:
        swap_analysis(cfg, tables, console)
    console.print(f"done: {tables}/shortcut_*")
    return 0


# --------------------------------------------------------------------------------------------------------------------
# 2. Leakage audit
# --------------------------------------------------------------------------------------------------------------------
# Image leakage audit between roles (`python main.py leakaudit`, CPU): exact duplicates of the decoded pixels and
# perceptual near-duplicates (64-bit pHash and dHash, the query in its 8 dihedral views) between every pair of
# processed cohorts. A pair counts as a near-duplicate at pHash distance <= PHASH_MAX and dHash distance <= DHASH_MAX.
# Single cells on black (C-NMC, auxiliary cells) hash mostly their black background, so near-duplicate hits between
# those two cohorts are reported but not meaningful. Writes results_dir/tables/leak_audit.{json,md}.

LEAK_COHORTS = {"train (C-NMC)": "train", "auxiliary (Bodzas)": "train_aux_bodzas2023",
                "selection (LeukemiaAttri)": "val", "ALL-IDB2": "test_allidb2", "Aria": "test_aria",
                "replicate (L_100X_C1)": "confirm_l100x"}
BLACK_COHORTS = {"train (C-NMC)", "auxiliary (Bodzas)"}
PHASH_MAX, DHASH_MAX = 6, 10
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def bits_to_uint64(bits: np.ndarray) -> np.uint64:
    return np.uint64(int("".join("1" if b else "0" for b in bits.ravel()), 2))


def hashes(grey: np.ndarray) -> tuple:
    """pHash (DCT of 32 x 32, 8 x 8 low frequencies without DC) and dHash (9 x 8 horizontal gradient)."""
    img = Image.fromarray(grey)
    d = dctn(np.asarray(img.resize((32, 32), Image.BILINEAR), dtype=np.float64), norm="ortho")[:8, :8].ravel()
    p = np.r_[d[1:], d[0]] > np.median(d[1:])
    small = np.asarray(img.resize((9, 8), Image.BILINEAR), dtype=np.float64)
    return bits_to_uint64(p), bits_to_uint64(small[:, 1:] > small[:, :-1])


def cohort_hashes(root: str) -> Dict[str, np.ndarray]:
    files = sorted(os.path.join(root, c, f) for c in ("Normal", "ALL") if os.path.isdir(os.path.join(root, c))
                   for f in os.listdir(os.path.join(root, c)))
    md5, ph, dh = [], [], []
    for path in files:
        rgb = np.asarray(Image.open(path).convert("RGB"))
        md5.append(hashlib.md5(rgb.tobytes()).hexdigest())
        grey = np.asarray(Image.fromarray(rgb).convert("L"))
        views = [np.rot90(grey, k)[:, ::-1] if flip else np.rot90(grey, k) for k in range(4) for flip in (0, 1)]
        hp, hd = zip(*(hashes(np.ascontiguousarray(v)) for v in views))
        ph.append(hp)
        dh.append(hd)
    return {"files": np.array(files), "md5": np.array(md5), "phash": np.array(ph, dtype=np.uint64),
            "dhash": np.array(dh, dtype=np.uint64)}


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.bitwise_xor(a[:, None], b[None, :])
    return POPCOUNT[x.view(np.uint8).reshape(*x.shape, 8)].sum(-1)


def near_pairs(query: Dict, ref: Dict, chunk: int = 512) -> tuple:
    """Near-duplicate pairs (query in any of its 8 views vs reference view 0) and the smallest pHash distance."""
    hits, best = 0, 64
    for s in range(0, len(query["phash"]), chunk):
        qp, qd = query["phash"][s:s + chunk], query["dhash"][s:s + chunk]
        dp = np.min([hamming(qp[:, v], ref["phash"][:, 0]) for v in range(8)], axis=0)
        dd = np.min([hamming(qd[:, v], ref["dhash"][:, 0]) for v in range(8)], axis=0)
        hits += int(((dp <= PHASH_MAX) & (dd <= DHASH_MAX)).sum())
        best = min(best, int(dp.min()))
    return hits, best


def leak_audit_main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.parse_args(argv)
    cfg = load_config()
    root = abs_path(cfg["phase1"]["data_dir"])
    data = {name: cohort_hashes(os.path.join(root, d)) for name, d in LEAK_COHORTS.items()
            if os.path.isdir(os.path.join(root, d))}
    rows = []
    for a, b in itertools.combinations(data, 2):
        exact = len(set(data[a]["md5"]) & set(data[b]["md5"]))
        hits, best = near_pairs(data[a], data[b])
        rows.append({"cohort_a": f"{a} ({len(data[a]['files'])})", "cohort_b": f"{b} ({len(data[b]['files'])})",
                     "exact_duplicates": exact, "near_duplicate_pairs": hits, "min_phash_distance": best,
                     "note": "black backgrounds dominate the hash" if {a, b} <= BLACK_COHORTS else ""})
        print(rows[-1])
    caption = (f"Image leakage audit between cohorts: exact duplicates of the decoded pixels and near-duplicate pairs "
               f"(pHash <= {PHASH_MAX} and dHash <= {DHASH_MAX}, query in its 8 dihedral views).")
    export_results_table(rows, abs_path(os.path.join(cfg["results_dir"], "tables", "leak_audit")), caption)
    return 0


# --------------------------------------------------------------------------------------------------------------------
# 3. Whole-field (Aria) inference audit
# --------------------------------------------------------------------------------------------------------------------
# Audit of the whole-field (Aria) inference (`python main.py fieldaudit [--only NAME ...]`); exploratory and post hoc
# (the tests are unblinded), no method choice follows from it.
#
# 1. detector: per Aria field, white cells detected before and after the whole-cell filters (`inference.cells`), mean
#    kept radius, fraction filtered, white-cell mask fraction and the two fallbacks (the filter would empty the field,
#    so every detection is kept; no detection at all, so the field is scored whole). Coverage per class and subtype,
#    and a detector-only reference: a logistic regression on these features, 5-fold GroupKFold over field groups.
# 2. field_scoring: Aria ROC-AUC of the five main arms' checkpoints with whole-field resize vs the per-cell pipeline
#    (mean, top-3), one view each, computed on this machine; the stored A100 8-view numbers are listed apart and never
#    compared with them.
# 3. bodzas_colour_rule: a logistic regression on in-cell colour statistics (Lab mean and sd, HSV saturation; central
#    Azure-B cell mask) fitted on the auxiliary Bodzas cells only, applied unchanged to the detected Aria cells (crops
#    at the inference scale, mean per field), ALL-IDB2 and the val cohort.
# 4. subtype_batches: grouped-CV ROC-AUC of the Aria background-only features (`shortcut.BACKGROUND`) for each pair
#    of subtypes.
# Writes results_dir/tables/field_audit_<name>.{json,md}; per-field intermediates are cached in --cache.

FIELD_ANALYSES = ("detector", "field_scoring", "bodzas_colour_rule", "subtype_batches")
MAIN_ARMS = ("basic", "hed_jitter", "randstainna", "stain_mixup", "morpho_mix")
SUBTYPES = ("Benign", "Early", "Pre", "Pro")
DETECTOR = ["n_detected", "n_kept", "frac_filtered", "mean_radius", "mask_frac", "fallback_kept_all",
            "scored_whole"]
COLOUR = ["L", "a", "b", "L_sd", "a_sd", "b_sd", "sat"]
MIN_MASK_PX = 20
AUX_DIR = "train_aux_bodzas2023"


def write_table(tables: str, name: str, caption: str, rows: List[Dict], payload: Dict) -> None:
    """field_audit_{name}.json (payload) and .md (bold caption and the rows as a markdown table)."""
    with open(os.path.join(tables, f"field_audit_{name}.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    cols = list(rows[0])
    cell = (lambda v: f"{v:.3f}" if isinstance(v, float) else str(v))
    lines = [f"**{caption}**", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(cell(r[c]) for c in cols) + " |" for r in rows]
    with open(os.path.join(tables, f"field_audit_{name}.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def aria_fields(cfg: Dict) -> Tuple[List[str], np.ndarray, np.ndarray, List[np.ndarray]]:
    """Names, labels (ALL = 1), subtypes and uint8 RGB fields of the Aria test cohort."""
    files, y = cohort_images(cfg, "aria")
    names = [os.path.basename(p) for p in files]
    subtype = np.array([n.split("_")[1] for n in names])
    return names, y, subtype, [np.asarray(Image.open(p).convert("RGB")) for p in files]


def whole_cell_flags(rgb: np.ndarray, cells: np.ndarray, min_inside: float, min_nucleus: float
                     ) -> Tuple[np.ndarray, float]:
    """The whole-cell test of `detect_cells` for each of its unfiltered detections, and the white-cell mask
    fraction of the field; mirrors detect_cells, which only returns the cells it keeps."""
    mask = white_cell_mask(rgb).astype(bool)
    nucleus = extract_blast_cell_prior(rgb)[1].astype(bool) & mask
    yy, xx = np.mgrid[:rgb.shape[0], :rgb.shape[1]]
    flags = []
    for x, y, r in cells.astype(np.float64):
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        inside = (d2 <= (1.5 * r) ** 2).sum() / (np.pi * (1.5 * r) ** 2)
        flags.append(inside >= min_inside and nucleus[d2 <= r * r].mean() >= min_nucleus)
    return np.asarray(flags, dtype=bool), float(mask.mean())


def split_rows(flat: np.ndarray, counts: np.ndarray) -> List[np.ndarray]:
    return np.split(flat, np.cumsum(counts)[:-1])


def detections(cfg: Dict, names: List[str], fields: List[np.ndarray], cache: str) -> Dict:
    """Per field: unfiltered detections, the configured (filtered, with fallback) detections, the whole-cell flags
    of the unfiltered ones and the white-cell mask fraction; cached."""
    path = os.path.join(cache, "detections.npz")
    if not os.path.exists(path):
        s = cfg["inference"]["cells"]
        r, inside, nucleus = float(s["min_radius"]), float(s["min_inside"]), float(s["min_nucleus"])
        det, kept, flags, frac = [], [], [], []
        for f in fields:
            det.append(detect_cells(f, r))
            kept.append(detect_cells(f, r, min_inside=inside, min_nucleus=nucleus))
            fl, mf = whole_cell_flags(f, det[-1], inside, nucleus)
            flags.append(fl)
            frac.append(mf)
        np.savez(path, names=np.array(names), det=np.concatenate(det), det_n=[len(d) for d in det],
                 kept=np.concatenate(kept), kept_n=[len(k) for k in kept], flags=np.concatenate(flags),
                 mask_frac=np.asarray(frac))
    z = np.load(path)
    if list(z["names"]) != names:
        raise RuntimeError(f"{path} was built for other fields; delete it")
    return {"det": split_rows(z["det"], z["det_n"]), "kept": split_rows(z["kept"], z["kept_n"]),
            "flags": split_rows(z["flags"], z["det_n"]), "mask_frac": z["mask_frac"]}


def detector_features(d: Dict) -> pd.DataFrame:
    rows = []
    for det, kept, flags, mf in zip(d["det"], d["kept"], d["flags"], d["mask_frac"]):
        rows.append({"n_detected": len(det), "n_kept": len(kept),
                     "frac_filtered": 1 - flags.mean() if len(det) else 0.0,
                     "mean_radius": float(kept[:, 2].mean()) if len(kept) else 0.0, "mask_frac": float(mf),
                     "fallback_kept_all": int(len(det) > 0 and not flags.any()), "scored_whole": int(len(kept) == 0),
                     "filter_consistent": int(len(kept) == (int(flags.sum()) or len(det)))})
    return pd.DataFrame(rows)


def coverage_row(name: str, f: pd.DataFrame) -> Dict:
    q1, med, q3 = np.percentile(f["n_kept"], [25, 50, 75])
    return {"subset": name, "fields": len(f), "kept median [IQR]": f"{med:.0f} [{q1:.0f}-{q3:.0f}]",
            "detected median": float(f["n_detected"].median()), "frac filtered mean": float(f["frac_filtered"].mean()),
            "kept radius median": float(f.loc[f["n_kept"] > 0, "mean_radius"].median()),
            "fallback kept all": int(f["fallback_kept_all"].sum()), "scored whole": int(f["scored_whole"].sum())}


def detector_audit(cfg: Dict, data: Tuple, d: Dict, tables: str, console) -> None:
    names, y, subtype, _ = data
    f = detector_features(d)
    groups = groups_for(cfg, "aria", names)
    x = StandardScaler().fit_transform(f[DETECTOR].values.astype(np.float64))
    reference = cv_auc(x, y, groups)
    single = {c: round(float(roc_auc_score(y, f[c])), 4) for c in DETECTOR}
    rows = [coverage_row(c, f[y == k]) for k, c in ((0, "Normal"), (1, "ALL"))]
    rows += [coverage_row(s, f[subtype == s]) for s in SUBTYPES]
    payload = {"settings": cfg["inference"]["cells"], "features": DETECTOR, "detector_only_cv_auc": reference,
               "single_feature_auc_all_positive": single, "coverage": rows,
               "filter_mismatch_fields": int((f["filter_consistent"] == 0).sum()),
               "per_field": {"names": names, **{c: f[c].tolist() for c in DETECTOR}}}
    caption = (f"Aria detector coverage per class and subtype (kept = after the whole-cell filters); a logistic "
               f"regression on the {len(DETECTOR)} per-field detector features alone reaches grouped-CV ROC-AUC "
               f"{reference:.3f} (single-feature AUCs in the JSON).")
    write_table(tables, "detector", caption, rows, payload)
    console.print(f"detector: LR {reference:.3f}; single " + ", ".join(f"{k} {v:.3f}" for k, v in single.items()))


@torch.no_grad()
def resize_scores(model: torch.nn.Module, fields: Sequence[np.ndarray], img_size: int, views: int,
                  device: torch.device, autocast, batch_size: int) -> np.ndarray:
    """p(ALL) of every field resized to img_size (`get_val_transforms` on the GPU)."""
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    out = []
    for s in range(0, len(fields), batch_size):
        x = torch.from_numpy(np.stack(fields[s:s + batch_size])).to(device).permute(0, 3, 1, 2).float() / 255
        x = torch.nn.functional.interpolate(x, size=(img_size, img_size), mode="bilinear", antialias=True)
        with autocast() if autocast is not None else torch.autocast(device.type, enabled=False):
            out.append(tta_logits(model, (x - mean) / std, n_views=views).float().exp()[:, 1].cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def run_scores(cfg: Dict, path: str, fields: List[np.ndarray], kept: List[np.ndarray], views: int,
               device: torch.device, batch_size: int, cache_file: str) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Resize p(ALL) per field and per-cell p(ALL) of the kept cells for one checkpoint, `views` views; cached."""
    if not os.path.exists(cache_file):
        model = load_phase1_model(cfg, path, device)
        autocast = autocast_for(cfg["mixed_precision"], device) if device.type == "cuda" else None
        resize = resize_scores(model, fields, cfg["img_size"], views, device, autocast, batch_size)
        cells = score_crops(model, fields, kept, float(cfg["inference"]["cells"]["scale"]), cfg["img_size"], views,
                            device, batch_size=batch_size, autocast=autocast)
        np.savez(cache_file, resize=resize, cells=np.concatenate(cells), n=[len(c) for c in cells])
        del model
        torch.cuda.empty_cache()
    z = np.load(cache_file)
    return z["resize"], split_rows(z["cells"], z["n"])


def stored_a100(cfg: Dict, arm: str, seed: int, names: List[str], y: np.ndarray, top_k: int) -> Dict:
    """The reported A100 Aria ROC-AUC (8 views, mean) and the top-k one from the stored per-cell scores."""
    with open(abs_path(os.path.join(cfg["results_dir"], "tables", "extended_metrics_runs.json")),
              encoding="utf-8") as fh:
        reported = {(r["arm"], r["seed"]): r["roc_auc"] for r in json.load(fh) if r["cohort"] == "aria"}
    pred_path = abs_path(os.path.join(cfg["results_dir"], "predictions", f"phase1_{arm}_seed{seed}_test_aria.json"))
    with open(pred_path, encoding="utf-8") as fh:
        pred = json.load(fh)
    if pred["names"] != names or not np.array_equal(pred["y_true"], y):
        raise RuntimeError(f"{pred_path} does not list the Aria fields in cohort order")
    cells = [np.asarray(c, dtype=np.float64) for c in pred["cell_probs"]]
    top = field_scores(cells, np.asarray(pred["probs"])[:, 1], "topk", top_k)
    return {"a100_8view_mean": reported.get((arm, seed)), "a100_8view_top3": float(roc_auc_score(y, top))}


def field_scoring_audit(cfg: Dict, data: Tuple, d: Dict, tables: str, cache: str, seeds: Optional[List[int]],
                        batch_size: int, console) -> None:
    names, y, subtype, fields = data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    top_k = int(cfg["inference"]["cells"]["top_k"])
    ckpt_dir = abs_path(os.path.join(cfg["results_dir"], "checkpoints"))
    runs, device_check = [], None
    for arm in MAIN_ARMS:
        for path in sorted(glob.glob(os.path.join(ckpt_dir, f"phase1_{arm}_seed*_best.pt"))):
            seed = int(CKPT.search(os.path.basename(path))["seed"])
            if seeds and seed not in seeds:
                continue
            resize, cells = run_scores(cfg, path, fields, d["kept"], 1, device, batch_size,
                                       os.path.join(cache, f"scores_{arm}_seed{seed}_1view.npz"))
            row = {"arm": arm, "seed": seed, "resize_1view": float(roc_auc_score(y, resize)),
                   "cells_mean_1view": float(roc_auc_score(y, field_scores(cells, resize, "mean"))),
                   f"cells_top{top_k}_1view": float(roc_auc_score(y, field_scores(cells, resize, "topk", top_k))),
                   "cells_max_1view": float(roc_auc_score(y, field_scores(cells, resize, "max")))}
            runs.append({**row, **stored_a100(cfg, arm, seed, names, y, top_k)})
            console.print("  " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                           for k, v in runs[-1].items()))
            if arm == "morpho_mix" and device_check is None:  # device drift: local 8 views vs the stored A100 run
                r8, c8 = run_scores(cfg, path, fields, d["kept"], 8, device, batch_size,
                                    os.path.join(cache, f"scores_{arm}_seed{seed}_8view.npz"))
                local = field_scores(c8, r8, "mean")
                pred = os.path.join(cfg["results_dir"], "predictions", f"phase1_{arm}_seed{seed}_test_aria.json")
                with open(abs_path(pred), encoding="utf-8") as fh:
                    dp = np.abs(local - np.asarray(json.load(fh)["probs"])[:, 1])
                device_check = {"arm": arm, "seed": seed, "local_8view_mean_auc": float(roc_auc_score(y, local)),
                                "a100_8view_mean_auc": runs[-1]["a100_8view_mean"], "max_abs_dp": float(dp.max()),
                                "mean_abs_dp": float(dp.mean())}
    df = pd.DataFrame(runs)
    cols = [c for c in df.columns if c not in ("arm", "seed")]
    rows = []
    for arm, g in df.groupby("arm", sort=False):
        sd = (lambda c: f" +/- {g[c].std(ddof=1):.3f}" if len(g) > 1 else "")
        rows.append({"arm": arm, "seeds": len(g), **{c: f"{g[c].mean():.3f}{sd(c)}" for c in cols}})
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    payload = {"device": name, "views_local": 1,
               "note": "local columns (this device, 1 view) are compared only with each other; a100_* columns are "
                       "the stored Colab A100 runs (8 views), listed for reference",
               "device_check": device_check, "runs": runs}
    caption = ("Aria ROC-AUC (mean +/- sd over seeds) of whole-field resize vs per-cell scoring, 1 view on this "
               "machine; the A100 8-view columns are the stored Colab runs, for reference only.")
    write_table(tables, "field_scoring", caption, rows, payload)


def colour_features(rgb: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    """Lab mean and sd and HSV saturation mean inside `mask`; None when the mask is (nearly) empty."""
    m = mask > 0
    if m.sum() < MIN_MASK_PX:
        return None
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)[m]
    sat = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1].astype(np.float32)[m]
    return np.r_[lab.mean(0), lab.std(0), sat.mean()]


def single_cell_colours(cfg: Dict, subdir: str, cache: str) -> pd.DataFrame:
    """Colour features of the central cell of every image of data_dir/subdir/{Normal,ALL}; cached."""
    path = os.path.join(cache, f"colour_{subdir}.csv")
    if not os.path.exists(path):
        tau, rows = float(cfg["morpho_threshold"]), []
        for y, cls in enumerate(cfg["phase1"]["class_names"]):
            for p in sorted(glob.glob(abs_path(os.path.join(cfg["phase1"]["data_dir"], subdir, cls, "*.png")))):
                rgb = np.asarray(Image.open(p).convert("RGB"))
                feat = colour_features(rgb, central_cell_mask(rgb, tau))
                rows.append({"name": os.path.basename(p), "y": y,
                             **dict(zip(COLOUR, feat if feat is not None else [np.nan] * len(COLOUR)))})
        pd.DataFrame(rows).to_csv(path, index=False)
    return pd.read_csv(path)


def aria_cell_colours(cfg: Dict, fields: List[np.ndarray], kept: List[np.ndarray], cache: str) -> pd.DataFrame:
    """Colour features of every kept Aria cell, cropped as at inference (224 / scale field px, resampled to
    img_size) with its central Azure-B cell mask; cached."""
    path = os.path.join(cache, "colour_aria_cells.csv")
    if not os.path.exists(path):
        tau, size = float(cfg["morpho_threshold"]), int(cfg["img_size"])
        window = size / float(cfg["inference"]["cells"]["scale"])
        rows = []
        for i, (f, cells) in enumerate(zip(fields, kept)):
            if not len(cells):
                continue
            field = torch.from_numpy(f).permute(2, 0, 1)[None].float() / 255
            crops = crop_cells(field, torch.zeros(len(cells), dtype=torch.long), torch.from_numpy(cells[:, :2]),
                               window, size)
            for crop in (crops * 255).round().byte().permute(0, 2, 3, 1).numpy():
                feat = colour_features(crop, central_cell_mask(crop, tau))
                rows.append({"field": i, **dict(zip(COLOUR, feat if feat is not None else [np.nan] * len(COLOUR)))})
        pd.DataFrame(rows).to_csv(path, index=False)
    return pd.read_csv(path)


def bodzas_colour_audit(cfg: Dict, data: Tuple, d: Dict, tables: str, cache: str, console) -> None:
    names, y, subtype, fields = data
    aux_all = single_cell_colours(cfg, AUX_DIR, cache)
    aux = aux_all.dropna()
    rule = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(aux[COLOUR].values, aux["y"])
    cv = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), aux[COLOUR].values,
                           aux["y"], cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")
    aux_dropped = len(aux_all) - len(aux)
    rows = [{"cohort": "Bodzas aux (fit, in-sample)", "n": len(aux), "dropped (empty mask)": aux_dropped,
             "roc_auc": float(roc_auc_score(aux["y"], rule.predict_proba(aux[COLOUR].values)[:, 1]))},
            {"cohort": "Bodzas aux (5-fold CV, ungrouped)", "n": len(aux), "dropped (empty mask)": aux_dropped,
             "roc_auc": float(roc_auc_score(aux["y"], cv[:, 1]))}]
    cells = aria_cell_colours(cfg, fields, d["kept"], cache)
    valid = cells.dropna()
    valid = valid.assign(p=rule.predict_proba(valid[COLOUR].values)[:, 1])
    per_field = valid.groupby("field")["p"].mean()
    idx = per_field.index.values
    ya, sa, pa = y[idx], subtype[idx], per_field.values
    dropped = len(fields) - len(idx)
    rows.append({"cohort": "Aria fields (ALL vs Benign)", "n": len(idx), "dropped (empty mask)": dropped,
                 "roc_auc": float(roc_auc_score(ya, pa))})
    for s in SUBTYPES[1:]:
        keep = (sa == s) | (sa == "Benign")
        rows.append({"cohort": f"Aria fields ({s} vs Benign)", "n": int(keep.sum()), "dropped (empty mask)": "",
                     "roc_auc": float(roc_auc_score(ya[keep], pa[keep]))})
    for key, subdir in (("ALL-IDB2", "test_allidb2"), ("val (LeukemiaAttri)", "val")):
        f = single_cell_colours(cfg, subdir, cache)
        ok = f.dropna()
        rows.append({"cohort": key, "n": len(ok), "dropped (empty mask)": len(f) - len(ok),
                     "roc_auc": float(roc_auc_score(ok["y"], rule.predict_proba(ok[COLOUR].values)[:, 1]))})
    lr = rule[-1]
    payload = {"features": COLOUR, "mask": "central_cell_mask at morpho_threshold (Aria: on the inference crops)",
               "standardised_coef": dict(zip(COLOUR, lr.coef_[0].round(4).tolist())),
               "intercept": float(lr.intercept_[0]), "aux_empty_mask_dropped": aux_dropped,
               "aria_cells": int(len(cells)), "aria_cells_empty_mask": int(len(cells) - len(valid)),
               "aria_fields_without_kept_cells": int(sum(len(k) == 0 for k in d["kept"])), "rows": rows}
    caption = ("A colour-only rule fitted on the Bodzas auxiliary cells (in-cell Lab mean/sd and saturation, "
               "ALL vs Normal) and applied unchanged to the test and val cohorts; ROC-AUC, ALL = positive.")
    write_table(tables, "bodzas_colour_rule", caption, rows, payload)
    console.print("bodzas rule: " + ", ".join(f"{r['cohort']} {r['roc_auc']:.3f}" for r in rows))


def subtype_batch_audit(cfg: Dict, tables: str, console) -> None:
    f = pd.read_csv(os.path.join(tables, "shortcut_features_aria.csv"))
    sub = f["name"].str.split("_").str[1].values
    groups = groups_for(cfg, "aria", f["name"])
    pairs = [("Early", "Pre"), ("Early", "Pro"), ("Pre", "Pro"), ("Benign", "Early"), ("Benign", "Pre"),
             ("Benign", "Pro"), ("Benign", "ALL")]
    rows = []
    for a, b in pairs:
        keep = (sub == a) | ((sub != "Benign") if b == "ALL" else (sub == b))
        x = StandardScaler().fit_transform(f.loc[keep, BACKGROUND].values)
        yy = (sub[keep] != a).astype(int)
        rows.append({"pair (negative vs positive)": f"{a} vs {b}", "fields": int(keep.sum()),
                     "groups": int(len(set(groups[keep]))), "background_only_cv_auc": cv_auc(x, yy, groups[keep])})
    caption = ("Grouped 5-fold CV ROC-AUC of a logistic regression on the Aria background-only features for each "
               "pair of subtypes; values near 1 mean the subtypes are separable from the background alone.")
    write_table(tables, "subtype_batches", caption, rows, {"features": BACKGROUND, "rows": rows})
    console.print("subtype batches: " + ", ".join(f"{r['pair (negative vs positive)']} "
                                                  f"{r['background_only_cv_auc']:.3f}" for r in rows))


def field_audit_main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", choices=FIELD_ANALYSES, default=list(FIELD_ANALYSES))
    p.add_argument("--seeds", nargs="+", type=int, help="field_scoring seeds (default: every checkpoint)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--cache", default="results/cache/field_audit")
    a = p.parse_args(argv)
    cfg = load_config()
    console = get_console()
    tables, cache = abs_path(os.path.join(cfg["results_dir"], "tables")), abs_path(a.cache)
    os.makedirs(cache, exist_ok=True)
    data = d = None
    if set(a.only) - {"subtype_batches"}:
        data = aria_fields(cfg)
        d = detections(cfg, data[0], data[3], cache)
    if "detector" in a.only:
        detector_audit(cfg, data, d, tables, console)
    if "subtype_batches" in a.only:
        subtype_batch_audit(cfg, tables, console)
    if "bodzas_colour_rule" in a.only:
        bodzas_colour_audit(cfg, data, d, tables, cache, console)
    if "field_scoring" in a.only:
        field_scoring_audit(cfg, data, d, tables, cache, a.seeds, a.batch_size, console)
    console.print(f"done: {tables}/field_audit_*")
    return 0


# --------------------------------------------------------------------------------------------------------------------
# 4. Pre-specified replicate (LeukemiaAttri L_100X_C1)
# --------------------------------------------------------------------------------------------------------------------
# Pre-registered confirmatory cohort (`python main.py confirm`): LeukemiaAttri L_100X_C1, the selection cohort's
# laboratory imaged with its low-cost microscope, scored once by every stored checkpoint (docs/decision_log.md,
# entry of 2026-10-07). It likely shares slides with the selection cohort, so it is an acquisition-shift replicate,
# not a new population.
#
# Stages: `build` crops the cohort with the selection-cohort rules (`leukemiaattri.build_val_cohort`, unchanged);
# `score` writes results_dir/confirm_l100x/predictions/{experiment id}_confirm.json for every checkpoint (8-view TTA)
# and for the two DinoBloom probes, refitted with their stored C (their weights were not saved); `analyze` writes
# results_dir/tables/confirm_l100x_*: per-arm metrics, MorphoMix against the six comparators (the hierarchical
# bootstrap of `method_level_tests`, slides as units) and the background-only reference.

RAW = "data/raw/LeukemiaAttri/L_100X_C1"
COHORT = "data/processed/confirm_l100x"
KEY = "confirm"
H_FIELD = (640, 640)  # the selection subset's squeezed field size, where x_scale 0.60 was measured
N_BOOT = 2000


def build(console) -> pd.DataFrame:
    fields = sorted(glob.glob(os.path.join(abs_path(RAW), "**", "*.png"), recursive=True))
    sizes = {Image.open(p).size for p in fields[:50]}
    x_scale = 0.60 if sizes == {H_FIELD} else 1.0  # the pre-registered rule
    console.print(f"{len(fields)} fields, sizes {sorted(sizes)} -> x_scale {x_scale}")
    return build_val_cohort(abs_path(RAW), abs_path(COHORT), x_scale=x_scale)


def out_dir(cfg: Dict) -> str:
    return abs_path(os.path.join(cfg["results_dir"], "confirm_l100x"))


def score(cfg: Dict, console) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast = autocast_for(cfg["mixed_precision"], device)
    loader = eval_loader(BinaryLeukemiaDataset(abs_path(COHORT), get_val_transforms(cfg["img_size"])), cfg)
    args = dict(tta_views=cfg.get("tta_views", 8), device=device, autocast=autocast,
                channels_last=cfg["channels_last"])
    root = out_dir(cfg)
    done = lambda eid: os.path.exists(os.path.join(root, "predictions", f"{eid}_{KEY}.json"))  # noqa: E731
    for path in sorted(glob.glob(abs_path(os.path.join(cfg["results_dir"], "checkpoints", "phase1_*_best.pt")))):
        eid = os.path.basename(path)[:-len("_best.pt")]
        if done(eid):
            continue
        model = load_phase1_model(cfg, path, device)
        train._save_predictions(root, eid, KEY, predict_images(model, loader, **args))
        console.print(f"  {eid}")
        del model
        torch.cuda.empty_cache()
    for name in foundation.MODELS:
        eid = f"phase1_{name}_seed{cfg['primary_seed']}"
        if done(eid):
            continue
        stored = abs_path(os.path.join(cfg["results_dir"], "tables", f"foundation_{name}.json"))
        with open(stored, encoding="utf-8") as f:
            c = float(json.load(f)["c"])
        setup_run(cfg, int(cfg["primary_seed"]))
        backbone = foundation.timm.create_model(foundation.MODELS[name], pretrained=True, num_classes=0).eval()
        dims = backbone.num_features
        dummy = foundation.ProbeModel(backbone, StandardScaler().fit(np.zeros((2, dims))),
                                      LogisticRegression().fit(np.zeros((2, dims)), [0, 1])).to(device).eval()
        extra = [abs_path(cfg["aux_train"]["dir"])] if cfg["aux_train"]["enabled"] else []
        fit_ds = BinaryLeukemiaDataset(os.path.join(abs_path(cfg["phase1"]["data_dir"]), train.TRAIN_DIR),
                                       get_val_transforms(cfg["img_size"]), extra_dirs=extra)
        x_tr, y_tr = foundation.extract(dummy, eval_loader(fit_ds, cfg), device, autocast)
        scaler = StandardScaler().fit(x_tr)
        probe = LogisticRegression(C=c, class_weight="balanced", max_iter=5000).fit(scaler.transform(x_tr), y_tr)
        model = foundation.ProbeModel(backbone, scaler, probe).to(device).eval()
        train._save_predictions(root, eid, KEY, predict_images(model, loader, **args))
        console.print(f"  {eid} (refitted, C={c:g})")
        del model, backbone, dummy
        torch.cuda.empty_cache()


def slide_groups(cfg: Dict, cohort: str, names: List[str]) -> np.ndarray:
    manifest = pd.read_csv(os.path.join(abs_path(COHORT), "manifest.csv"))
    slide = {os.path.basename(f): str(s) for f, s in zip(manifest["file"], manifest["slide"])}
    return np.array([slide[os.path.basename(n)] for n in names])


def analyze(cfg: Dict, console) -> None:
    preds = {}
    for path in sorted(glob.glob(os.path.join(out_dir(cfg), "predictions", f"phase1_*_{KEY}.json"))):
        arm, seed = re.match(rf"phase1_(.+)_seed(\d+)_{KEY}\.json$", os.path.basename(path)).groups()
        with open(path, encoding="utf-8") as f:
            preds[(KEY, arm, int(seed))] = json.load(f)
    tables = abs_path(os.path.join(cfg["results_dir"], "tables"))
    rng = np.random.default_rng(int(cfg["primary_seed"]))
    rows = []
    for (_, arm, seed), d in sorted(preds.items()):
        y, p = np.asarray(d["y_true"]), np.asarray(d["probs"], dtype=np.float64)[:, 1]
        m = cohort_metrics(y, p, 0.5, slide_groups(cfg, KEY, d["names"]), 200, rng)
        rows.append({"arm": arm, "seed": seed, **{k: round(float(v), 4) for k, v in m.items()}})
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(tables, "confirm_l100x_runs.csv"), index=False)
    summary = runs.groupby("arm").agg(seeds=("seed", "size"), roc_auc=("roc_auc", "mean"),
                                      roc_auc_sd=("roc_auc", "std"), auprc=("auprc", "mean"),
                                      macro_f1=("macro_f1", "mean"), positive_rate=("predicted_positive_rate", "mean"),
                                      ece=("ece", "mean")).round(4).reset_index()
    main = [*foundation.MODELS, *cfg["augmentations"]]
    main_preds = {k: v for k, v in preds.items() if k[1] in main}
    tests = method_level_tests(cfg, main_preds, "morpho_mix", N_BOOT, np.random.default_rng(0), groups_fn=slide_groups)
    ablation_preds = {k: v for k, v in preds.items()
                      if k[1].startswith("abl") or (k[1] == "morpho_mix" and k[2] in cfg["ablation_seeds"])}
    ablation = method_level_tests(cfg, ablation_preds, "morpho_mix", N_BOOT, np.random.default_rng(0),
                                  groups_fn=slide_groups)

    first = next(iter(preds.values()))
    names, y = first["names"], np.asarray(first["y_true"])
    files = [os.path.join(abs_path(COHORT), ("Normal", "ALL")[c], n) for c, n in zip(y, names)]
    tau = float(cfg["morpho_threshold"])
    feats = np.array([background_features(np.asarray(Image.open(p).convert("RGB")), False, tau) for p in files])
    x = StandardScaler().fit_transform(feats[:, :len(BACKGROUND)])
    groups = slide_groups(cfg, KEY, names)
    reference, n_slides = cv_auc(x, y, groups), len(set(groups))
    index = {n: i for i, n in enumerate(names)}
    residual = []
    for (_, arm, seed), d in sorted(preds.items()):
        order = [index[n] for n in d["names"]]
        p = np.clip(np.asarray(d["probs"], dtype=np.float64)[:, 1], 1e-6, 1 - 1e-6)
        s = np.log(p / (1 - p))
        design = np.c_[np.ones(len(order)), x[order]]
        resid = s - design @ np.linalg.lstsq(design, s, rcond=None)[0]
        yy = np.asarray(d["y_true"])
        residual.append({"arm": arm, "seed": seed, "auc": roc_auc_score(yy, s),
                         "auc_residual": roc_auc_score(yy, resid)})
    pd.DataFrame(residual).to_csv(os.path.join(tables, "confirm_l100x_residual_runs.csv"), index=False)

    caption = (f"Pre-registered confirmatory cohort LeukemiaAttri L_100X_C1 ({len(y)} crops, {int(y.sum())} ALL, "
               f"{n_slides} slides; likely the selection cohort's slides under another microscope). Per arm: mean "
               f"over seeds, 8-view TTA, scored on one local device. Background-only reference (grouped CV by slide): "
               f"ROC-AUC {reference:.3f}.")
    export_results_table(summary.to_dict("records"), os.path.join(tables, "confirm_l100x_summary"), caption)
    boot = "hierarchical bootstrap over slides within class and seeds (2,000 draws)"
    export_results_table(tests, os.path.join(tables, "confirm_l100x_tests"),
                         f"MorphoMix minus each comparator on L_100X_C1 (pre-registered primary analysis): seed-mean "
                         f"ROC-AUC / AUPRC, {boot}, Holm over six.")
    export_results_table(ablation, os.path.join(tables, "confirm_l100x_ablation"),
                         f"Arm a (MorphoMix, seeds 42-44) minus arms b-k on L_100X_C1 (descriptive), {boot}.")
    with open(os.path.join(tables, "confirm_l100x_background.json"), "w", encoding="utf-8") as f:
        json.dump({"background_only_roc_auc": reference, "n": int(len(y)), "n_all": int(y.sum()),
                   "n_slides": n_slides}, f, indent=2)
    console.print(summary.to_string(index=False))
    console.print(pd.DataFrame(tests).query("metric == 'roc_auc'").to_string(index=False))
    console.print(f"background-only ROC-AUC {reference:.3f}")


def confirm_main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stages", nargs="*", default=["build", "score", "analyze"], choices=["build", "score", "analyze"])
    a = p.parse_args(argv)
    cfg = load_config()
    console = get_console()
    for stage in a.stages:
        {"build": lambda: build(console), "score": lambda: score(cfg, console),
         "analyze": lambda: analyze(cfg, console)}[stage]()
    return 0
