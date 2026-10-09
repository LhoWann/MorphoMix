"""Classification metrics (binary Normal vs ALL, and any number of classes)."""
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

DEGENERATE_RATE = 0.95  # one-class rate reported as degenerate


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str]
) -> Dict[str, Any]:
    """Macro and per-class F1, precision, recall and specificity, balanced accuracy, kappa, confusion matrix."""
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    n_classes = len(class_names)

    macro_f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0, labels=list(range(n_classes))))
    bal_acc = float(balanced_accuracy_score(y_t, y_p))
    acc = float(accuracy_score(y_t, y_p))
    macro_prec = float(precision_score(y_t, y_p, average="macro", zero_division=0, labels=list(range(n_classes))))
    macro_rec = float(recall_score(y_t, y_p, average="macro", zero_division=0, labels=list(range(n_classes))))
    kappa = float(cohen_kappa_score(y_t, y_p))

    per_class_f1_vals = f1_score(y_t, y_p, average=None, zero_division=0, labels=list(range(n_classes)))
    per_class_prec_vals = precision_score(y_t, y_p, average=None, zero_division=0, labels=list(range(n_classes)))
    per_class_rec_vals = recall_score(y_t, y_p, average=None, zero_division=0, labels=list(range(n_classes)))

    per_class_f1 = {name: float(val) for name, val in zip(class_names, per_class_f1_vals)}
    per_class_prec = {name: float(val) for name, val in zip(class_names, per_class_prec_vals)}
    per_class_rec = {name: float(val) for name, val in zip(class_names, per_class_rec_vals)}

    cm = confusion_matrix(y_t, y_p, labels=list(range(n_classes)))

    # one-vs-rest specificity per class
    per_class_spec = {}
    specificities = []
    for c, name in enumerate(class_names):
        tp = cm[c, c]
        fn = sum(cm[c, :]) - tp
        fp = sum(cm[:, c]) - tp
        tn = cm.sum() - tp - fn - fp
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0
        per_class_spec[name] = spec
        specificities.append(spec)
    macro_spec = float(np.mean(specificities))

    return {
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
        "accuracy": acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_specificity": macro_spec,
        "cohen_kappa": kappa,
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_prec,
        "per_class_recall": per_class_rec,
        "per_class_specificity": per_class_spec,
        "confusion_matrix": cm.tolist()
    }


def threshold_free_metrics(
    y_true: List[int],
    probs_positive: List[float]
) -> Dict[str, Any]:
    """ROC-AUC and AUPRC for the binary cohorts."""
    y = np.asarray(y_true)
    p = np.asarray(probs_positive, dtype=np.float64)
    if y.size == 0 or len(np.unique(y)) < 2 or not np.isfinite(p).all():  # non-finite: a diverged model
        return {"roc_auc": float("nan"), "auprc": float("nan")}
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
    }


def probability_diagnostics(
    y_true: List[int],
    probs_positive: List[float],
    y_pred: List[int],
    positive_index: int = 1
) -> Dict[str, Any]:
    """Mandatory companions to every cross-cohort number."""
    y = np.asarray(y_true)
    p = np.asarray(probs_positive, dtype=np.float64)
    pred = np.asarray(y_pred)
    rate = float((pred == positive_index).mean()) if pred.size else float("nan")
    return {
        "predicted_positive_rate": rate,
        "mean_p_positive_on_negatives": float(p[y == 0].mean()) if (y == 0).any() else float("nan"),
        "mean_p_positive_on_positives": float(p[y == 1].mean()) if (y == 1).any() else float("nan"),
        "degenerate": bool(rate >= DEGENERATE_RATE or rate <= 1.0 - DEGENERATE_RATE),
    }
