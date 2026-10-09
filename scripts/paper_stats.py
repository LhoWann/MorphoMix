"""Post-hoc paper statistics from stored predictions, training logs and shortcut tables (`python main.py paperstats`).

No model is loaded; every analysis reads results_dir/{predictions, logs, tables} and writes
results_dir/tables/paper_<name>.{md,json} (calibration also paper_reliability_bins.csv). All of it is exploratory
(the tests are unblinded); bootstraps resample units within class as `analysis` does (Aria field groups, ALL-IDB2
cell groups, val slides) unless a table says otherwise. Analyses (`--only`):
  val_curves         last-epoch and all-epoch-mean val ROC-AUC per arm, Welch tests, selected epochs
  seed_t_tests       Welch / one-sample t over seeds, a sensitivity check of `method_level_tests`
  rank_correlation   Spearman rank agreement of ROC-AUC between cohorts over 17 configurations
  aria_subtypes      Aria Early / Pre / Pro vs Benign ROC-AUC with field-group bootstrap CIs
  pseudo_patient     Aria method-level bootstrap with contiguous pseudo-patient blocks as units
  normal_auprc       AUPRC with Normal as the positive class
  calibration        Brier score, Brier skill score, ECE and seed-ensemble reliability bins
  residual_controls  permutation and background-only negative controls of the residual AUC (`audit.shortcut_main`)
  swap_summary       background-swap effects with t-intervals, no-aux minus aux pairs, background removal
  ablation_tests     hierarchical bootstrap of arm a vs b-g and of every with- vs without-aux pair
  thresholds         decision metrics at 0.5, at the val Youden threshold and after an EM prior-shift correction
  temperature        Brier score, Brier skill score and ECE before and after val-fitted temperature scaling"""
import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "2")  # a CPU-only post-hoc job, run next to other work

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
from typing import Callable, Dict, List, Optional, Sequence, Tuple  # noqa: E402

from src.utils.config import abs_path, load_config, setup_cuda_env  # noqa: E402
setup_cuda_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402
from scipy.special import expit  # noqa: E402
from scipy.stats import rankdata, spearmanr, t as t_dist  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss, matthews_corrcoef, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold, StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from scripts.audit import BACKGROUND, cv_auc, feature_table, groups_for  # noqa: E402
from scripts.evaluate import (  # noqa: E402
    ECE_BINS, FOUNDATION_ARMS, PRED_FILE, analysis_groups, expected_calibration_error, holm, resample_units,
    unit_members,
)
from scripts.experiments import ARM_AUG, ARMS, REFERENCE_ARM  # noqa: E402
from src.evaluation.calibration import DECISION_THRESHOLD, resolve_threshold  # noqa: E402
from src.evaluation.metrics import DEGENERATE_RATE  # noqa: E402
from src.utils.logger import get_console  # noqa: E402

REFERENCE = ARM_AUG
BASELINES = ("basic", "hed_jitter", "randstainna", "stain_mixup")
MAIN = (*BASELINES, REFERENCE)
ABLATION = tuple(f"abl{k}" for k in ARMS if k != REFERENCE_ARM)
PROBES = FOUNDATION_ARMS
ARM_ORDER = (*MAIN, *ABLATION, *PROBES)
COMPONENT_ARMS = tuple(f"abl{k}" for k, v in ARMS.items() if k != REFERENCE_ARM and v["aug"] == REFERENCE
                       and v["cfg"] != {"aux_train.enabled": False})
NO_AUX_PAIRS = sorted(((v["aug"], f"abl{k}") for k, v in ARMS.items() if v["cfg"] == {"aux_train.enabled": False}),
                      key=lambda pair: MAIN.index(pair[0]))  # (arm with the auxiliary cells, the same without)
COHORTS = ("val", "allidb2", "aria")
TEST_COHORTS = COHORTS[1:]
DECISION_RULES = (DECISION_THRESHOLD, "val_youden", "prior_shift")
DECISION_METRICS = ("threshold", "balanced_accuracy", "sensitivity", "specificity", "mcc", "positive_rate")
SUBTYPES = ("Early", "Pre", "Pro")
# Aria has 89 patients without identifiers: per class, contiguous runs of the numbered fields stand in for them
PSEUDO_PATIENTS = {"89 pseudo-patients (25/23/22/19)": {"Benign": 25, "Early": 23, "Pre": 22, "Pro": 19},
                   "45 blocks (13/12/11/9)": {"Benign": 13, "Early": 12, "Pre": 11, "Pro": 9}}
N_PERMUTATIONS = 100
LOG_T_BOUND = 5.0  # temperature search range e^-5 .. e^5
P_COLUMN = re.compile(r"(^|_)p(_holm)?$")
Section = Tuple[str, str, List[Dict]]  # (key, caption, rows)


class Cohort:
    """Every run's p(ALL) on one cohort, aligned to one image order (sorted file names)."""

    def __init__(self, names: List[str], y: np.ndarray, probs: Dict[Tuple[str, int], np.ndarray]):
        self.names = names
        self.y = y
        self.probs = probs

    def seeds(self, arm: str) -> List[int]:
        return sorted(s for a, s in self.probs if a == arm)

    def stack(self, arm: str, seeds: Optional[Sequence[int]] = None) -> np.ndarray:
        """[seeds, images] p(ALL); a single-run arm (a frozen probe) gives its one run whatever `seeds` asks for."""
        own = self.seeds(arm)
        use = [s for s in own if s in seeds] if seeds is not None and len(own) > 1 else own
        return np.stack([self.probs[(arm, s)] for s in use])


class Context:
    """Config, paths and the lazily loaded predictions shared by the analyses."""

    def __init__(self, cfg: Dict, n_boot: int):
        self.cfg = cfg
        self.n_boot = n_boot
        self.results = abs_path(cfg["results_dir"])
        self.tables = os.path.join(self.results, "tables")
        self.short_seeds = list(cfg["ablation_seeds"])
        self._cohorts: Optional[Dict[str, Cohort]] = None

    @property
    def cohorts(self) -> Dict[str, Cohort]:
        if self._cohorts is None:
            self._cohorts = load_cohorts(os.path.join(self.results, "predictions"))
        return self._cohorts


def load_cohorts(pred_dir: str) -> Dict[str, Cohort]:
    raw: Dict[str, Dict[Tuple[str, int], Dict]] = {}
    for path in sorted(glob.glob(os.path.join(pred_dir, "phase1_*_seed*_*.json"))):
        m = PRED_FILE.match(os.path.basename(path))
        if m and m["aug"] in ARM_ORDER:
            with open(path, encoding="utf-8") as f:
                raw.setdefault(m["cohort"], {})[(m["aug"], int(m["seed"]))] = json.load(f)
    out = {}
    for cohort, runs in raw.items():
        names = sorted(os.path.basename(n) for n in next(iter(runs.values()))["names"])
        pos = {n: i for i, n in enumerate(names)}
        y, probs = None, {}
        for key, d in runs.items():
            order = np.array([pos[os.path.basename(n)] for n in d["names"]])
            if len(order) != len(names):
                raise RuntimeError(f"{cohort} {key}: {len(order)} images, expected {len(names)}")
            yy, p = np.empty(len(names), dtype=np.int64), np.empty(len(names))
            yy[order], p[order] = d["y_true"], np.asarray(d["probs"], dtype=np.float64)[:, 1]
            if y is not None and not np.array_equal(y, yy):
                raise RuntimeError(f"{cohort} {key}: labels differ from the other runs")
            y, probs[key] = yy, p
        out[cohort] = Cohort(names, y, probs)
    return out


# --- statistics ---

def auc_rows(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """ROC-AUC of every row of `p` (Mann-Whitney, ties averaged as sklearn does)."""
    pos = int(y.sum())
    neg = y.size - pos
    return (rankdata(p, axis=-1)[..., y == 1].sum(-1) - pos * (pos + 1) / 2) / (pos * neg)


def t_test(a: Sequence[float], b) -> Dict[str, float]:
    """Welch's t of mean(a) - mean(b) for a sample `b`, a one-sample t against the value `b` for a scalar; with the
    95 % t-interval of the difference."""
    a = np.asarray(a, dtype=np.float64)
    va = a.var(ddof=1) / a.size
    if np.isscalar(b):
        delta, se, df = a.mean() - b, np.sqrt(va), a.size - 1.0
    else:
        b = np.asarray(b, dtype=np.float64)
        vb = b.var(ddof=1) / b.size
        delta, se = a.mean() - b.mean(), np.sqrt(va + vb)
        df = (va + vb) ** 2 / (va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1))
    t = delta / se
    half = t_dist.ppf(0.975, df) * se
    return {"delta": delta, "ci_low": delta - half, "ci_high": delta + half, "t": t, "df": df,
            "p": float(2 * t_dist.sf(abs(t), df))}


def t_interval(values: Sequence[float]) -> Tuple[float, float, float]:
    v = np.asarray(values, dtype=np.float64)
    half = t_dist.ppf(0.975, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else np.nan
    return float(v.mean()), float(v.mean() - half), float(v.mean() + half)


def add_holm(rows: List[Dict], key: str = "p") -> None:
    for r, adj in zip(rows, holm([r[key] for r in rows])):
        r[f"{key}_holm"] = adj


def hierarchical_deltas(y: np.ndarray, strata: List[List[np.ndarray]], runs: Dict[str, np.ndarray],
                        pairs: List[Tuple[str, str]], n_boot: int, rng: np.random.Generator) -> List[Dict]:
    """ROC-AUC of seed-mean differences (first minus second arm of each pair) with the hierarchical bootstrap of
    `evaluate.method_level_tests`: units resampled within each stratum (shared by every arm), then, independently
    per arm, its seeds. Two-sided p floored at 1/(B+1)."""
    arms = list(runs)
    start = np.cumsum([0] + [len(runs[a]) for a in arms])
    stacked = np.concatenate([runs[a] for a in arms])
    observed = {a: float(auc_rows(y, runs[a]).mean()) for a in arms}
    deltas = np.empty((n_boot, len(pairs)))
    for i in range(n_boot):
        idx = resample_units(strata, rng)
        auc = auc_rows(y[idx], stacked[:, idx])
        draw = {a: auc[start[j] + rng.choice(len(runs[a]), len(runs[a]))].mean() for j, a in enumerate(arms)}
        deltas[i] = [draw[r] - draw[c] for r, c in pairs]
    out = []
    for (r, c), d in zip(pairs, deltas.T):
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        out.append({"reference": r, "comparator": c, "reference_seeds": len(runs[r]),
                    "comparator_seeds": len(runs[c]), "reference_auc": observed[r], "comparator_auc": observed[c],
                    "delta": observed[r] - observed[c], "ci_low": float(np.percentile(d, 2.5)),
                    "ci_high": float(np.percentile(d, 97.5)), "p": max(p, 1.0 / (n_boot + 1))})
    return out


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)  # as shortcut.residual_analysis
    return np.log(p / (1 - p))


def decision_values(y: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, float]:
    """DECISION_METRICS of "ALL iff p(ALL) >= threshold", with the one-class flag of `metrics.DEGENERATE_RATE`."""
    pred = (p >= threshold).astype(np.int64)
    sens, spec, rate = pred[y == 1].mean(), 1 - pred[y == 0].mean(), pred.mean()
    return {"threshold": threshold, "balanced_accuracy": (sens + spec) / 2, "sensitivity": sens,
            "specificity": spec, "mcc": matthews_corrcoef(y, pred), "positive_rate": rate,
            "degenerate": int(rate >= DEGENERATE_RATE or rate <= 1 - DEGENERATE_RATE)}


def fit_temperature(y: np.ndarray, z: np.ndarray) -> float:
    """The T > 0 minimising the NLL of sigmoid(z / T), searched over |log T| <= LOG_T_BOUND."""
    def nll(log_t: float) -> float:
        s = z / np.exp(log_t)
        return float(np.mean(np.logaddexp(0, s) - y * s))

    return float(np.exp(minimize_scalar(nll, bounds=(-LOG_T_BOUND, LOG_T_BOUND), method="bounded").x))


def ensemble_bins(p: np.ndarray) -> np.ndarray:
    """Equal-width bin of each p(ALL), as `evaluate.expected_calibration_error`."""
    return np.clip(np.digitize(p, np.linspace(0, 1, ECE_BINS + 1)[1:-1]), 0, ECE_BINS - 1)


def sd(v: np.ndarray) -> float:
    return float(np.std(v, ddof=1)) if len(v) > 1 else np.nan


def by_arm(rows: List[Dict]) -> List[Dict]:
    return sorted(rows, key=lambda r: ARM_ORDER.index(r["arm"]) if r["arm"] in ARM_ORDER else len(ARM_ORDER))


def describe(arm: str) -> str:
    return ARMS[arm[3:]]["desc"] if arm.startswith("abl") else arm


# --- analyses ---

def val_curves(ctx: Context) -> List[Section]:
    log = pd.read_csv(os.path.join(ctx.results, "logs", "training_log.csv"))
    log = log[log["experiment_id"].str.match(r"^phase1_[a-z_]+_seed\d+$")]
    log = log.drop_duplicates(["experiment_id", "epoch"], keep="last").sort_values(["experiment_id", "epoch"])
    keys = ["val_roc_auc"] + (["val_loss"] if "val_loss" in log else [])
    runs = []
    for eid, c in log.groupby("experiment_id"):
        arm, seed = re.match(r"phase1_(.+)_seed(\d+)$", eid).groups()
        best = c.sort_values([*keys, "epoch"], ascending=[False, *[True] * len(keys)]).iloc[0]
        runs.append({"arm": arm, "seed": int(seed), "last": c["val_roc_auc"].iloc[-1],
                     "mean": c["val_roc_auc"].mean(), "epochs": len(c), "selected_epoch": int(best["epoch"]),
                     "logged_best_epoch": int(c.loc[c["is_best"] == 1, "epoch"].max())})
    runs = pd.DataFrame(runs)
    mismatch = runs[runs["selected_epoch"] != runs["logged_best_epoch"]]
    groups = [(a, a, runs[runs["arm"] == a]) for a in (*MAIN, *ABLATION)]
    groups.insert(len(MAIN), (f"a ({REFERENCE}, seeds {ctx.short_seeds[0]}-{ctx.short_seeds[-1]})", REFERENCE,
                              runs[(runs["arm"] == REFERENCE) & runs["seed"].isin(ctx.short_seeds)]))
    summary = []
    for label, arm, g in groups:
        if g.empty:
            continue
        sel = g["selected_epoch"]
        summary.append({"arm": label, "description": describe(arm), "seeds": len(g), "epochs": int(g["epochs"].max()),
                        "last_mean": g["last"].mean(), "last_sd": g["last"].std(ddof=1),
                        "all_epoch_mean": g["mean"].mean(), "all_epoch_sd": g["mean"].std(ddof=1),
                        "selected_epoch_min": int(sel.min()), "selected_epoch_median": float(sel.median()),
                        "selected_epoch_max": int(sel.max())})
    tests = []
    ref = runs[runs["arm"] == REFERENCE]
    for stat in ("last", "mean"):
        family = []
        for b in BASELINES:
            r = t_test(ref[stat], runs.loc[runs["arm"] == b, stat])
            family.append({"statistic": "last epoch" if stat == "last" else "mean over epochs", "comparator": b,
                           "morpho_mix": ref[stat].mean(), "comparator_mean": runs.loc[runs["arm"] == b, stat].mean(),
                           **r})
        add_holm(family)
        tests += family
    per_run = [{**r, "selection_differs_from_log": int(r["selected_epoch"] != r["logged_best_epoch"])}
               for r in runs.to_dict("records")]
    return [("summary", "Val (selection cohort) ROC-AUC without checkpoint selection, from the training log: mean and "
                        "sd over seeds of the last-epoch value and of the mean over all epochs; selected epoch = best "
                        "val ROC-AUC (ties: lower val loss), min / median / max over seeds.", summary),
            ("tests", "MorphoMix minus each baseline over seeds (5 vs 5), Welch t with 95 % CI of the difference; "
                      "Holm over the four baselines per statistic.", tests),
            ("runs", f"Per-run values ({len(mismatch)} runs whose selected epoch differs from the logged is_best "
                     "epoch).", per_run)]


def seed_t_tests(ctx: Context) -> List[Section]:
    boot = {}
    path = os.path.join(ctx.tables, "method_level_tests.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            boot = {(r["cohort"], r["metric"], r["comparator"]): r["p_holm"] for r in json.load(f)}
    rows = []
    for cohort in COHORTS:
        coh = ctx.cohorts[cohort]
        for metric, fn in (("roc_auc", roc_auc_score), ("auprc", average_precision_score)):
            value = {a: [fn(coh.y, p) for p in coh.stack(a)] for a in (*MAIN, *PROBES)}
            family = []
            for comp in (*BASELINES, *PROBES):
                single = len(value[comp]) == 1
                r = t_test(value[REFERENCE], value[comp][0] if single else value[comp])
                family.append({"cohort": cohort, "metric": metric, "comparator": comp,
                               "test": "one-sample t" if single else "Welch t",
                               "morpho_mix": float(np.mean(value[REFERENCE])),
                               "comparator_value": float(np.mean(value[comp])), **r})
            add_holm(family)
            for r in family:
                r["bootstrap_p_holm"] = boot.get((cohort, metric, r["comparator"]), np.nan)
            rows += family
    return [("tests", "Sensitivity of the method-level bootstrap: MorphoMix (5 seeds) minus each augmentation (Welch t "
                      "over the 5 seed values) and each frozen probe (one-sample t of the MorphoMix seeds against the "
                      "probe's single value), 95 % t-interval of the difference; Holm over the six comparators per "
                      "cohort and metric; bootstrap_p_holm from method_level_tests. Seeds are the only unit, the "
                      "test images are held fixed.", rows)]


def rank_correlation(ctx: Context) -> List[Section]:
    configs = []
    for arm in ARM_ORDER:
        means = {c: float(auc_rows(ctx.cohorts[c].y, ctx.cohorts[c].stack(arm, ctx.short_seeds)).mean())
                 for c in COHORTS}
        configs.append({"arm": arm, "description": describe(arm),
                        "seeds": len(ctx.cohorts["val"].stack(arm, ctx.short_seeds)), **means})
    df = pd.DataFrame(configs)
    corr = []
    for a, b in (("val", "allidb2"), ("val", "aria"), ("allidb2", "aria")):
        r = spearmanr(df[a], df[b])
        corr.append({"cohorts": f"{a} vs {b}", "configurations": len(df), "spearman_rho": float(r.statistic),
                     "p": float(r.pvalue)})
    return [("spearman", f"Spearman rank correlation of seed-mean ROC-AUC between cohorts over {len(df)} "
                         f"configurations ({len(df) - len(PROBES)} trained arms on seeds {ctx.short_seeds}, "
                         f"{len(PROBES)} frozen probes). Descriptive: the configurations share components and are "
                         "not independent, so p is nominal.", corr),
            ("configurations", "Seed-mean ROC-AUC per configuration.", configs)]


def aria_subtypes(ctx: Context) -> List[Section]:
    coh = ctx.cohorts["aria"]
    subtype = np.array([n.split("_")[1] for n in coh.names])
    groups = analysis_groups(ctx.cfg, "aria", coh.names)
    arms = [a for a in ARM_ORDER if coh.seeds(a)]
    start = np.cumsum([0] + [len(coh.seeds(a)) for a in arms])
    stacked = np.concatenate([coh.stack(a) for a in arms])
    rows = [{"arm": a, "seeds": len(coh.seeds(a)),
             "benign_median_p_all": float(np.median(stacked[start[j]:start[j + 1]][:, subtype == "Benign"], 1).mean())}
            for j, a in enumerate(arms)]
    rng = np.random.default_rng(0)
    for st in SUBTYPES:
        keep = (subtype == "Benign") | (subtype == st)
        y, p = coh.y[keep], stacked[:, keep]
        strata = unit_members(y, groups[keep])
        draws = (resample_units(strata, rng) for _ in range(ctx.n_boot))
        boot = np.stack([auc_rows(y[idx], p[:, idx]) for idx in draws])
        observed = auc_rows(y, p)
        for j, r in enumerate(rows):
            seg = slice(start[j], start[j + 1])
            mean = boot[:, seg].mean(1)
            r.update({st.lower(): float(observed[seg].mean()), f"{st.lower()}_ci_low": float(np.percentile(mean, 2.5)),
                      f"{st.lower()}_ci_high": float(np.percentile(mean, 97.5))})
    n = ", ".join(f"{st} {int((subtype == st).sum())}" for st in ("Benign", *SUBTYPES))
    return [("subtypes", f"Aria ROC-AUC of each B-ALL subtype against Benign (fields: {n}), mean over the seeds of "
                         f"each arm, with a 95 % percentile CI of that seed mean from {ctx.n_boot:,} bootstrap draws "
                         "over field groups within class (seeds held fixed); benign_median_p_all is the seed-mean "
                         "median p(ALL) of the Benign fields. Descriptive.", rows)]


def pseudo_patient(ctx: Context) -> List[Section]:
    coh = ctx.cohorts["aria"]
    subtype = np.array([n.split("_")[1] for n in coh.names])
    number = np.array([int(re.findall(r"(\d+)\.png", n)[0]) for n in coh.names])
    runs = {a: coh.stack(a) for a in (REFERENCE, *BASELINES)}
    field = {}
    path = os.path.join(ctx.tables, "method_level_tests.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            field = {r["comparator"]: r for r in json.load(f) if r["cohort"] == "aria" and r["metric"] == "roc_auc"}
    rng = np.random.default_rng(0)  # as the original check, so its numbers reproduce
    rows = []
    for label, blocks in PSEUDO_PATIENTS.items():
        strata = []
        for st, k in blocks.items():  # Benign first: one stratum per subtype, Normal before ALL
            idx = np.flatnonzero(subtype == st)
            strata.append(np.array_split(idx[np.argsort(number[idx])], k))
        family = hierarchical_deltas(coh.y, strata, runs, [(REFERENCE, b) for b in BASELINES], ctx.n_boot, rng)
        add_holm(family)
        for r in family:
            ref = field.get(r["comparator"], {})
            rows.append({"units": label, "comparator": r["comparator"], "delta": r["delta"], "ci_low": r["ci_low"],
                         "ci_high": r["ci_high"], "p": r["p"], "p_holm": r["p_holm"],
                         **{f"field_group_{k}": ref.get(k, np.nan) for k in ("ci_low", "ci_high", "p_holm")}})
    feats = pd.read_csv(os.path.join(ctx.tables, "shortcut_features_aria.csv"))
    sub = feats["name"].str.split("_").str[1].to_numpy()
    num = feats["name"].str.extract(r"(\d+)\.png")[0].astype(int).to_numpy()
    x = StandardScaler().fit_transform(feats[BACKGROUND].values)
    background = []
    for label, blocks in PSEUDO_PATIENTS.items():
        units = np.empty(len(feats), dtype=object)
        for st, k in blocks.items():
            idx = np.flatnonzero(sub == st)
            for b, part in enumerate(np.array_split(idx[np.argsort(num[idx])], k)):
                units[part] = f"{st}_{b}"
        background.append({"units": label, "background_only_roc_auc": cv_auc(x, feats["y"].to_numpy(), units)})
    return [("tests", "Aria MorphoMix minus each augmentation (seed-mean ROC-AUC, 5 seeds each) with the hierarchical "
                      f"bootstrap ({ctx.n_boot:,} draws) over contiguous blocks of the numbered fields within each "
                      "subtype (an assumed stand-in for the 89 patients, which have no identifiers) and over seeds; "
                      "Holm over the four augmentations. field_group_* repeats the method-level test with field "
                      "groups as units (Holm there over six comparators, probes included).", rows),
            ("background", "Aria background-only reference (shortcut.cv_auc, eight background features) with the same "
                           "pseudo-patient blocks as cross-validation groups instead of field groups.", background)]


def normal_auprc(ctx: Context) -> List[Section]:
    rows = []
    for cohort in COHORTS:
        coh = ctx.cohorts[cohort]
        for arm in ARM_ORDER:
            if not coh.seeds(arm):
                continue
            normal = [average_precision_score(1 - coh.y, 1 - p) for p in coh.stack(arm)]
            rows.append({"cohort": cohort, "arm": arm, "seeds": len(normal), "normal_prevalence": 1 - coh.y.mean(),
                         "auprc_normal": float(np.mean(normal)),
                         "auprc_normal_sd": sd(normal),
                         "auprc_all": float(np.mean([average_precision_score(coh.y, p) for p in coh.stack(arm)]))})
    return [("normal", "AUPRC with Normal as the positive class (score 1 - p(ALL)), mean and sd over seeds (a probe "
                       "has one run); normal_prevalence is its chance level; auprc_all repeats the ALL-positive "
                       "AUPRC.", rows)]


def calibration(ctx: Context) -> List[Section]:
    rows, bins = [], []
    for cohort in COHORTS:
        coh = ctx.cohorts[cohort]
        prev = coh.y.mean()
        for arm in ARM_ORDER:
            if not coh.seeds(arm):
                continue
            p = coh.stack(arm)
            brier = np.array([brier_score_loss(coh.y, q) for q in p])
            ece = np.array([expected_calibration_error(coh.y, q) for q in p])
            bss = 1 - brier / (prev * (1 - prev))
            rows.append({"cohort": cohort, "arm": arm, "seeds": len(p), "prevalence_all": prev,
                         "brier": brier.mean(), "brier_sd": sd(brier), "brier_skill": bss.mean(),
                         "brier_skill_sd": sd(bss), "ece": ece.mean(), "ece_sd": sd(ece)})
            ens = p.mean(0)
            which = ensemble_bins(ens)
            for b in range(ECE_BINS):
                m = which == b
                bins.append({"cohort": cohort, "arm": arm, "seeds": len(p), "bin": b, "bin_low": b / ECE_BINS,
                             "bin_high": (b + 1) / ECE_BINS, "mean_p": float(ens[m].mean()) if m.any() else np.nan,
                             "fraction_all": float(coh.y[m].mean()) if m.any() else np.nan, "count": int(m.sum())})
    pd.DataFrame(bins).to_csv(os.path.join(ctx.tables, "paper_reliability_bins.csv"), index=False)
    return [("calibration", f"Calibration of p(ALL), mean (sd) over seeds: Brier score, Brier skill score against the "
                            f"constant cohort-prevalence predictor (1 - Brier / (prev (1 - prev))), ECE over "
                            f"{ECE_BINS} equal-width bins. Reliability bins of each arm's seed ensemble (mean p(ALL)) "
                            "in paper_reliability_bins.csv.", rows)]


def background_oof(x: np.ndarray, y: np.ndarray, g: Optional[np.ndarray]) -> np.ndarray:
    """Out-of-fold p(ALL) of the label-trained background-only model, with the folds of `shortcut.cv_auc`."""
    folds = (GroupKFold(5).split(x, y, g) if g is not None
             else StratifiedKFold(5, shuffle=True, random_state=0).split(x, y))
    oof = np.zeros(len(y))
    for tr, te in folds:
        oof[te] = LogisticRegression(max_iter=2000).fit(x[tr], y[tr]).predict_proba(x[te])[:, 1]
    return oof


def residual_controls(ctx: Context) -> List[Section]:
    rows, reference = [], []
    for cohort in COHORTS:
        coh = ctx.cohorts[cohort]
        f = feature_table(ctx.cfg, cohort, ctx.tables)  # the cached shortcut_features_{cohort}.csv
        pos = {n: i for i, n in enumerate(f["name"])}
        take = np.array([pos[n] for n in coh.names])
        if not np.array_equal(f["y"].values[take], coh.y):
            raise RuntimeError(f"{cohort}: feature-table labels differ from the predictions")
        x = StandardScaler().fit_transform(f[BACKGROUND].values)[take]
        n = len(x)
        designs = [x] + [x[np.random.default_rng(k).permutation(n)] for k in range(N_PERMUTATIONS)]
        bases = [np.linalg.qr(np.c_[np.ones(n), d])[0] for d in designs]

        def residual_aucs(s: np.ndarray) -> np.ndarray:  # [real, permutations...]
            return auc_rows(coh.y, np.stack([s - q @ (q.T @ s) for q in bases]))

        for arm in (*MAIN, *PROBES):
            per_seed = []
            for p in coh.stack(arm):
                s = logit(p)
                res = residual_aucs(s)
                auc = float(auc_rows(coh.y, s))
                per_seed.append((auc, res[0], auc - res[1:]))
            drops = np.stack([d for *_, d in per_seed])
            rows.append({"cohort": cohort, "arm": arm, "seeds": len(per_seed),
                         "auc": float(np.mean([a for a, *_ in per_seed])),
                         "auc_residual": float(np.mean([r for _, r, _ in per_seed])),
                         "drop": float(np.mean([a - r for a, r, _ in per_seed])),
                         "permutation_drop_mean": float(drops.mean()),
                         "permutation_drop_p95": float(np.percentile(drops.mean(0), 95))})
        oof = background_oof(StandardScaler().fit_transform(f[BACKGROUND].values), f["y"].values,
                             groups_for(ctx.cfg, cohort, f["name"]))
        s = logit(oof[take])
        res = residual_aucs(s)
        q = bases[0]
        reference.append({"cohort": cohort, "background_only_auc": float(auc_rows(coh.y, s)),
                          "r2_on_background": float(1 - (s - q @ (q.T @ s)).var() / s.var()),
                          "background_only_residual_auc": float(res[0]),
                          "permutation_residual_auc_mean": float(res[1:].mean())})
    return [("permutation", f"Negative control of the residual ROC-AUC: logit p(ALL) of each run regressed on the "
                            f"background features ({', '.join(BACKGROUND)}) as in the shortcut analysis (drop = auc - "
                            f"auc_residual), and on the same features shuffled across images ({N_PERMUTATIONS} "
                            "permutations, seeds 0-99): permutation_drop_mean is the drop expected with no image-level "
                            "link between score and background (mean over seeds and permutations), "
                            "permutation_drop_p95 the 95th percentile of the seed-mean drop over permutations.", rows),
            ("background_only", "Calibration point: the out-of-fold logit of the label-trained background-only model "
                                "(5-fold, grouped as in shortcut.cv_auc), a score built from the background alone, "
                                "put through the same residual step. Each fold fits its own coefficients, so the score "
                                "is not one linear function of the features and the residual keeps the fold-to-fold "
                                "part: a residual AUC away from 0.5 here is what the linear step leaves of a pure "
                                "background score.", reference)]


def swap_summary(ctx: Context) -> List[Section]:
    df = pd.read_csv(os.path.join(ctx.tables, "shortcut_bg_swap_runs.csv"))
    cells = df[df["cohort"] == "allidb2"]
    effects = []
    for arm, g in cells.groupby("arm"):
        mean, lo, hi = t_interval(g["class_specific_effect"])
        effects.append({"arm": arm, "description": describe(arm), "seeds": len(g), "auc": g["auc"].mean(),
                        "auc_other": g["auc_other"].mean(), "auc_same": g["auc_same"].mean(),
                        "class_specific_effect": mean, "ci_low": lo, "ci_high": hi})
    effect = cells.set_index(["arm", "seed"])["class_specific_effect"]
    pairs, pooled = [], []
    for aux, no_aux in NO_AUX_PAIRS:
        d = [effect[(no_aux, s)] - effect[(aux, s)] for s in ctx.short_seeds]
        pooled += d
        mean, lo, hi = t_interval(d)
        pairs.append({"pair": f"{no_aux} - {aux}", "seeds": len(d), "mean_difference": mean, "ci_low": lo,
                      "ci_high": hi, "positive": f"{sum(v > 0 for v in d)}/{len(d)}",
                      "per_seed": ", ".join(f"{v:+.3f}" for v in d)})
    mean, lo, hi = t_interval(pooled)
    pairs.append({"pair": "all pairs", "seeds": len(pooled), "mean_difference": mean, "ci_low": lo, "ci_high": hi,
                  "positive": f"{sum(v > 0 for v in pooled)}/{len(pooled)}",
                  "per_seed": f"min {min(pooled):+.3f}, max {max(pooled):+.3f}"})
    removal = []
    for (cohort, arm), g in df.groupby(["cohort", "arm"]):
        removal.append({"cohort": cohort, "arm": arm, "seeds": len(g), "auc": g["auc"].mean(),
                        "auc_none": g["auc_none"].mean(), "difference": (g["auc_none"] - g["auc"]).mean()})
    removal = sorted(by_arm(removal), key=lambda r: r["cohort"])
    return [("effects", "ALL-IDB2 background swap: class-specific effect (auc_same - auc_other, single view, "
                        "mean over 3 donor draws) per arm, mean over seeds with a 95 % t-interval.", by_arm(effects)),
            ("no_aux_pairs", f"Paired by seed ({ctx.short_seeds}): class-specific effect without the auxiliary "
                             "cells minus with them (main arms restricted to the same seeds); mean, 95 % t-interval, "
                             "count of positive differences.", pairs),
            ("background_removal", "Background removal: single-view ROC-AUC with the original image (auc) and with "
                                   "the cell on black (auc_none), mean over seeds; difference = auc_none - auc.",
             removal)]


def ablation_tests(ctx: Context) -> List[Section]:
    rng = np.random.default_rng(0)
    families = {"components": [(REFERENCE, a) for a in COMPONENT_ARMS], "auxiliary cells": NO_AUX_PAIRS}
    rows = []
    for cohort in COHORTS:
        coh = ctx.cohorts[cohort]
        strata = unit_members(coh.y, analysis_groups(ctx.cfg, cohort, coh.names))
        runs = {a: coh.stack(a, ctx.short_seeds) for a in (*MAIN, *ABLATION)}
        pairs = [p for fam in families.values() for p in fam]
        result = dict(zip(pairs, hierarchical_deltas(coh.y, strata, runs, pairs, ctx.n_boot, rng)))
        for name, fam in families.items():
            family = [{"cohort": cohort, "family": name, "comparator_description": describe(c), **result[(r, c)]}
                      for r, c in fam]
            add_holm(family)
            rows += family
    return [("tests", f"Descriptive (3 seeds per arm, seeds {ctx.short_seeds}): seed-mean ROC-AUC of the reference "
                      "(MorphoMix = arm a, or the arm trained with the auxiliary cells) minus the comparator, with the "
                      f"hierarchical bootstrap of method_level_tests ({ctx.n_boot:,} draws over units within class "
                      "and seeds); Holm within each cohort and family (components: a vs b-g; auxiliary cells: the "
                      "five with- vs without-aux pairs).", rows)]


def thresholds(ctx: Context) -> List[Section]:
    val = ctx.cohorts["val"]
    rows, counts = [], []
    for cohort in TEST_COHORTS:
        coh = ctx.cohorts[cohort]
        flags = {rule: {"augmentation": [], "probe": []} for rule in DECISION_RULES}
        for arm in (*MAIN, *PROBES):
            for rule in DECISION_RULES:
                runs = []
                for s in coh.seeds(arm):
                    p = coh.probs[(arm, s)]
                    runs.append(decision_values(coh.y, p, resolve_threshold(rule, val.y, val.probs[(arm, s)], p)))
                runs = pd.DataFrame(runs)
                row = {"cohort": cohort, "arm": arm, "rule": str(rule), "seeds": len(runs),
                       "prevalence_all": coh.y.mean()}
                for k in DECISION_METRICS:
                    row.update({k: runs[k].mean(), f"{k}_sd": sd(runs[k])})
                row["degenerate"] = f"{runs['degenerate'].sum()}/{len(runs)}"
                rows.append(row)
                flags[rule]["probe" if arm in PROBES else "augmentation"] += runs["degenerate"].tolist()
        for rule, f in flags.items():
            runs = {f"degenerate_{kind}_runs": f"{sum(v)}/{len(v)}" for kind, v in f.items()}
            counts.append({"cohort": cohort, "rule": str(rule), **runs})
    return [("rules", f"Decision rules on the test cohorts, mean and sd over seeds (a probe has one run): ALL iff "
                      f"p(ALL) >= {DECISION_THRESHOLD}, >= the Youden threshold of the same run's val (selection "
                      "cohort) predictions, or >= 1 - pi with pi the test ALL prevalence re-estimated by EM on the "
                      "unlabelled test scores (Saerens et al. 2002, balanced training prior; "
                      "calibration.resolve_threshold); no rule reads a test label. degenerate = runs with a "
                      f"positive rate >= {DEGENERATE_RATE} or <= {1 - DEGENERATE_RATE:.2f}. Exploratory.", rows),
            ("degenerate", "Degenerate runs per rule: the five augmentation arms (5 seeds each) and the two frozen "
                           "probes.", counts)]


def temperature(ctx: Context) -> List[Section]:
    val = ctx.cohorts["val"]
    fitted = {key: fit_temperature(val.y, logit(p)) for key, p in val.probs.items() if key[0] in (*MAIN, *PROBES)}
    rows = []
    for cohort in COHORTS:
        coh = ctx.cohorts[cohort]
        prev = coh.y.mean()
        for arm in (*MAIN, *PROBES):
            t = np.array([fitted[(arm, s)] for s in coh.seeds(arm)])
            row = {"cohort": cohort, "arm": arm, "seeds": len(t), "prevalence_all": prev,
                   "temperature": t.mean(), "temperature_min": t.min(), "temperature_max": t.max(),
                   "at_bound": f"{int((np.log(t) > LOG_T_BOUND - 1e-3).sum())}/{len(t)}"}
            for stage, ps in (("before", coh.stack(arm)), ("after", expit(logit(coh.stack(arm)) / t[:, None]))):
                brier = np.array([brier_score_loss(coh.y, q) for q in ps])
                bss = 1 - brier / (prev * (1 - prev))
                row.update({f"brier_{stage}": brier.mean(), f"brier_skill_{stage}": bss.mean(),
                            f"ece_{stage}": np.mean([expected_calibration_error(coh.y, q) for q in ps])})
            rows.append(row)
    return [("temperature", "Temperature scaling of p(ALL): one T per run fitted on its val (selection cohort) logits "
                            "z = log(p / (1 - p)) by minimum NLL, applied as sigmoid(z / T) to the same run's test "
                            "predictions (val rows are in-sample). Brier score, Brier skill score against the constant "
                            f"cohort-prevalence predictor and ECE ({ECE_BINS} equal-width bins), mean over seeds, "
                            "before and after. T > 0 is monotonic and keeps logit 0 at p = 0.5, so ROC-AUC, AUPRC and "
                            "every decision at the 0.5 threshold are unchanged; only the probabilities move. It has no "
                            "intercept, so it cannot absorb a prior shift: at_bound counts runs whose T reached "
                            f"e^{LOG_T_BOUND:g}, i.e. the val NLL is lowest with every p(ALL) pulled to 0.5.", rows)]


ANALYSES: Dict[str, Callable[[Context], List[Section]]] = {
    "val_curves": val_curves, "seed_t_tests": seed_t_tests, "rank_correlation": rank_correlation,
    "aria_subtypes": aria_subtypes, "pseudo_patient": pseudo_patient, "normal_auprc": normal_auprc,
    "calibration": calibration, "residual_controls": residual_controls, "swap_summary": swap_summary,
    "ablation_tests": ablation_tests, "thresholds": thresholds, "temperature": temperature,
}


# --- output ---

def _fmt(column: str, v) -> str:
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else f"{v:.4f}" if P_COLUMN.search(column) else f"{v:.3f}"
    return str(v)


def _plain(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else round(float(v), 6)
    return v


def write_tables(tables: str, name: str, sections: List[Section]) -> str:
    """paper_<name>.md (per section a bold caption and a markdown table) and paper_<name>.json (every section)."""
    md, payload = [], {}
    for key, caption, rows in sections:
        cols = list(dict.fromkeys(c for r in rows for c in r))
        md += [f"**{caption}**", "", "| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
        md += ["| " + " | ".join(_fmt(c, r.get(c, "")) for c in cols) + " |" for r in rows] + [""]
        payload[key] = {"caption": caption, "rows": [{k: _plain(v) for k, v in r.items()} for r in rows]}
    prefix = os.path.join(tables, f"paper_{name}")
    with open(prefix + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(prefix + ".json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return prefix + ".md"


def limit_resources() -> None:
    torch.set_num_threads(2)
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except (ImportError, OSError):
        pass


def paper_stats_main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", choices=list(ANALYSES), help="analyses to run (default: all)")
    p.add_argument("--n-boot", type=int, default=2000)
    a = p.parse_args(argv)
    limit_resources()
    console = get_console()
    ctx = Context(load_config(), a.n_boot)
    for name in a.only or ANALYSES:
        path = write_tables(ctx.tables, name, ANALYSES[name](ctx))
        console.print(f"ok {name} -> {path}")
    return 0
