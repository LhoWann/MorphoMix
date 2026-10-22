"""One append-only CSV of per-epoch metrics for every run."""
import csv
import json
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path


class UnifiedCSVLogger:
    """Appends one row per epoch to logs/training_log.csv under the results directory."""
    HEADER = [
        "timestamp",
        "experiment_id",
        "model",
        "augmentation",
        "seed",
        "epoch",
        "total_epochs",
        "train_loss",
        "val_loss",
        "val_macro_f1",
        "val_roc_auc",
        "val_balanced_acc",
        "val_accuracy",
        "learning_rate",
        "is_best",
        "per_class_f1",  # JSON string of all classes
    ]

    def __init__(self, output_dir: str = "results", filename: str = "training_log.csv"):
        self.log_dir = Path(output_dir) / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / filename
        self._ensure_header()

    def _ensure_header(self) -> None:
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.HEADER)

    def log_epoch(
        self,
        experiment_id: str,
        model: str,
        augmentation: str,
        seed: int,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_loss: float,
        val_macro_f1: float,
        val_roc_auc: float,
        val_balanced_acc: float,
        val_accuracy: float,
        learning_rate: float = 0.0,
        is_best: bool = False,
        per_class_f1_dict: Optional[Dict[str, float]] = None
    ) -> None:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if per_class_f1_dict is None:
            per_class_f1_dict = {}
        class_str = json.dumps({k: round(v, 5) for k, v in per_class_f1_dict.items()})

        row = [
            now_str,
            experiment_id,
            model,
            augmentation,
            seed,
            epoch,
            total_epochs,
            f"{train_loss:.5f}",
            f"{val_loss:.5f}",
            f"{val_macro_f1:.5f}",
            f"{val_roc_auc:.5f}",
            f"{val_balanced_acc:.5f}",
            f"{val_accuracy:.5f}",
            f"{learning_rate:.2e}",
            1 if is_best else 0,
            class_str
        ]
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
