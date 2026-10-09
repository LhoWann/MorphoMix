"""Confusion matrix figures, one per split."""
import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from src.evaluation.metrics import DEGENERATE_RATE
from src.utils import figstyle


def degeneracy_note(cm: np.ndarray, class_names: List[str]) -> str:
    """Names the class that takes nearly every prediction, if one does."""
    total = int(cm.sum())
    if total == 0:
        return ""
    per_pred = cm.sum(axis=0)
    top = int(per_pred.argmax())
    rate = per_pred[top] / total
    return f"degenerate: {rate:.0%} predicted {class_names[top]}" if rate >= DEGENERATE_RATE else ""


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str, title: str) -> None:
    """Counts on a Blues map, true class on rows."""
    figstyle.apply()
    shown = np.asarray(cm)
    size = 1.0 + 0.55 * len(class_names)
    fig, ax = plt.subplots(figsize=(size + 0.6, size))
    im = ax.imshow(shown, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    threshold = shown.max() / 2 if shown.size else 0
    for i in range(shown.shape[0]):
        for j in range(shown.shape[1]):
            colour = "white" if shown[i, j] > threshold else "black"
            ax.text(j, i, f"{int(shown[i, j])}", ha="center", va="center", color=colour)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def save_confusion(cm: np.ndarray, class_names: List[str], save_path: str, heading: str) -> None:
    """Title = split and image count, plus a degeneracy note when one class takes >= 95 %."""
    cm = np.asarray(cm)
    note = degeneracy_note(cm, class_names)
    title = f"{heading} (n = {int(cm.sum())})" + (f"\n{note}" if note else "")
    plot_confusion_matrix(cm=cm, class_names=class_names, save_path=save_path, title=title)
