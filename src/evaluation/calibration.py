"""The decision rule for the binary cohorts and the label-free choices of its threshold."""
from typing import Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import roc_curve

from src.evaluation.metrics import compute_metrics, probability_diagnostics

DECISION_THRESHOLD = 0.5
THRESHOLD_RULES = ("val_youden", "prior_shift")


def youden_threshold(y_true: List[int], p_positive: List[float]) -> float:
    """The threshold that maximises sensitivity + specificity - 1 (Youden's J) on a labelled cohort."""
    fpr, tpr, thresholds = roc_curve(np.asarray(y_true), np.asarray(p_positive, dtype=np.float64))
    return float(min(1.0, thresholds[int(np.argmax(tpr - fpr))]))  # roc_curve's first threshold is +inf


def prior_shift_threshold(p_positive: List[float], train_prior: float = 0.5, iters: int = 1000,
                          tol: float = 1e-6) -> float:
    """Label-free threshold for a cohort whose class prior differs from training (Saerens et al., Neural Comput
    2002): the test prior pi is estimated by EM on the unlabelled posteriors, and "adjusted posterior >= 0.5" is
    "p >= 1 - pi" for a model trained at a balanced prior (class-weighted loss, `train_prior` 0.5)."""
    p = np.asarray(p_positive, dtype=np.float64)
    pi = train_prior
    for _ in range(iters):
        w1 = p * pi / train_prior
        q = w1 / (w1 + (1 - p) * (1 - pi) / (1 - train_prior))
        if abs(q.mean() - pi) < tol:
            break
        pi = float(q.mean())
    odds = (1 - pi) / pi * train_prior / (1 - train_prior)
    return float(odds / (1 + odds))


def resolve_threshold(rule: Union[float, str], val_y: Optional[List[int]] = None,
                      val_p: Optional[List[float]] = None, test_p: Optional[List[float]] = None) -> float:
    """config `decision_threshold`: a number, `val_youden` (Youden on the val cohort of the same checkpoint) or
    `prior_shift` (EM prior estimate on the unlabelled test scores). No rule reads a test label."""
    if rule == "val_youden":
        return youden_threshold(val_y, val_p)
    if rule == "prior_shift":
        return prior_shift_threshold(test_p)
    if isinstance(rule, str):
        raise ValueError(f"decision_threshold must be a number or one of {THRESHOLD_RULES}, got {rule!r}")
    return float(rule)


def decision_metrics(
    y_true: List[int],
    probs: List[List[float]],
    class_names: List[str],
    positive_index: int = 1,
    threshold: float = DECISION_THRESHOLD,
) -> Dict:
    """The metric bundle for "positive iff p(positive) >= threshold", plus probability diagnostics."""
    p_pos = np.asarray(probs, dtype=np.float64)[:, positive_index]
    y_pred = (p_pos >= threshold).astype(int).tolist()
    entry = compute_metrics(y_true, y_pred, class_names)
    entry.update(probability_diagnostics(y_true, p_pos.tolist(), y_pred))
    entry["threshold"] = float(threshold)
    entry["rule"] = f"p(positive) >= {threshold:.4g}"
    return entry
