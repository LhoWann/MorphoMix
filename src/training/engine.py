"""Trainer: the training loop shared by every arm, on Accelerate; MorphoMix augments the batch on the GPU."""
import os
import logging
import warnings
from typing import Dict, Any, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration

from src.augmentations.transforms import degrade_resolution
from src.training.losses import WeightedCrossEntropyLoss
from src.training.mixstyle import MixStyle
from src.training.scheduler import setup_optimizer
from rich.progress import Progress, TaskID
from src.evaluation.calibration import DECISION_THRESHOLD
from src.evaluation.metrics import compute_metrics, threshold_free_metrics
from src.evaluation.tta import tta_logits
from src.evaluation.confusion import save_confusion
from src.utils.logger import (
    VALIDATION_DESCRIPTION,
    create_progress_bar,
    epoch_description,
    format_metrics,
    get_console,
)
from src.utils.csv_logger import UnifiedCSVLogger
from src.utils.resources import ResourceMonitor

warnings.filterwarnings("ignore")
logging.getLogger("accelerate").setLevel(logging.ERROR)


MORPHO_ARMS = ("morpho_mix",)
STAIN_ARMS = ("hed_jitter", "randstainna", "stain_mixup")


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_names: List[str],
        augmentation: str = "basic",
        morpho_threshold: float = 0.6,
        style_bank: Optional[nn.Module] = None,
        appearance_prob: float = 0.0,
        appearance_alpha: tuple = (0.5, 1.0),
        virtual_template_prob: float = 0.0,
        acquisition_prob: float = 0.0,
        rsn_prob: float = 0.0,
        small_cell_prob: float = 0.0,
        small_cell_px: tuple = (24, 48),
        small_cell_lowdetail_prob: float = 0.5,
        field_prob: float = 0.0,
        field_cells: tuple = (2, 8),
        use_background: bool = True,
        background_prob: float = 0.5,
        background_rbc: tuple = (5, 25),
        field_rbc: tuple = (40, 120),
        feather_edges: bool = True,
        stain_prob: float = 0.5,
        hed_sigma: float = 0.05,
        randstainna_stats: Optional[Dict[str, Any]] = None,
        randstainna_std_hyper: float = -0.3,
        stain_bank: Optional[torch.Tensor] = None,
        ema_decay: float = 0.0,
        label_smoothing: float = 0.0,
        freeze_stages: int = 0,
        freeze_epochs: int = 0,
        probe_lr: float = 0.0,
        lowres_prob: float = 0.0,
        lowres_scale: tuple = (0.25, 0.75),
        mixstyle_p: float = 0.0,
        mixstyle_alpha: float = 0.1,
        mixstyle_stages: tuple = (0, 1),
        tta_views: int = 8,
        selection_metric: str = "macro_f1",
        epochs: int = 25,
        warmup_epochs: int = 2,
        lr: float = 1e-4,
        layer_decay: float = 0.9,
        weight_decay: float = 0.05,
        grad_accum_steps: int = 1,
        mixed_precision: str = "fp16",
        channels_last: bool = True,
        log_every: int = 5,
        experiment_id: str = "experiment",
        seed: int = 42,
        output_dir: str = "results",
        resources: Optional[Dict[str, Any]] = None,
        class_weights: Optional[torch.Tensor] = None
    ):
        self.model = model
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_names = list(class_names)
        self.num_classes = len(self.class_names)
        self.augmentation = augmentation.lower()
        self.morpho_threshold = morpho_threshold
        self.style_bank = style_bank  # C1 references (src.augmentations.style_bank.StyleBank), None = C1 off
        self.appearance_prob = float(appearance_prob)
        self.appearance_alpha = tuple(appearance_alpha)
        self.virtual_template_prob = float(virtual_template_prob)
        self.acquisition_prob = float(acquisition_prob)
        self.rsn_prob = float(rsn_prob)  # MorphoMix hybrid: RandStainNA instead of C1 (randstainna_stats)
        self.small_cell_prob = float(small_cell_prob)
        self.small_cell_px = tuple(small_cell_px)
        self.small_cell_lowdetail_prob = float(small_cell_lowdetail_prob)
        self.field_prob = float(field_prob)
        self.field_cells = tuple(field_cells)
        self.use_background = use_background
        self.background_prob = background_prob
        self.background_rbc = tuple(background_rbc)
        self.field_rbc = tuple(field_rbc)
        self.feather_edges = feather_edges
        self.stain_prob = float(stain_prob)
        self.hed_sigma = float(hed_sigma)
        self.randstainna_stats = randstainna_stats  # train-set template distribution (fit_randstainna)
        self.randstainna_std_hyper = float(randstainna_std_hyper)
        self.stain_bank = stain_bank  # MLL23 stain matrices [N, 3, 2] of the Stain Mix-up targets
        if self.augmentation == "randstainna" and randstainna_stats is None:
            raise ValueError("the randstainna arm needs randstainna_stats (data/processed/randstainna_stats.json)")
        if self.augmentation in MORPHO_ARMS and self.rsn_prob > 0 and randstainna_stats is None:
            raise ValueError("morpho_mix with rsn_prob > 0 needs randstainna_stats (as the randstainna arm)")
        if self.augmentation == "stain_mixup" and stain_bank is None:
            raise ValueError("the stain_mixup arm needs stain_bank (style bank key stain_matrix)")
        self.lowres_prob = float(lowres_prob)
        self.lowres_scale = tuple(lowres_scale)
        self.freeze_stages = int(freeze_stages)  # stem + the first `freeze_stages` stages get no gradient ...
        self.freeze_epochs = int(freeze_epochs)  # ... for this many epochs; 0 = the whole run
        if not 0 <= self.freeze_stages <= len(model.stages):
            raise ValueError(f"freeze_stages must be in 0..{len(model.stages)}, got {freeze_stages}")
        # LP-FT (Kumar et al., ICLR 2022): with probe_lr > 0 the frozen epochs are a linear probe at this constant lr,
        # and the warm-up + cosine schedule runs over the remaining epochs; 0 = the frozen epochs are part of it
        self.probe_lr = float(probe_lr)
        self.probe_epochs = self.freeze_epochs if self.probe_lr > 0 else 0
        if self.probe_lr > 0 and (self.freeze_stages != len(model.stages) or not 0 < self.freeze_epochs < epochs):
            raise ValueError(f"probe_lr needs freeze_stages {len(model.stages)} (the whole backbone) and 0 < "
                             f"freeze_epochs < epochs, got {freeze_stages} and {freeze_epochs}")
        self.probing = False
        self.tta_views = int(tta_views)
        if selection_metric not in ("macro_f1", "roc_auc"):
            raise ValueError(f"selection_metric must be macro_f1 or roc_auc, got {selection_metric!r}")
        if selection_metric == "roc_auc" and len(self.class_names) != 2:
            raise ValueError("selection_metric roc_auc needs a binary task")
        self.selection_metric = selection_metric
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.experiment_id = experiment_id
        self.seed = seed
        self.output_dir = output_dir
        self.channels_last = channels_last
        self.log_every = max(1, log_every)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=grad_accum_steps,
            mixed_precision=mixed_precision,
            dataloader_config=DataLoaderConfiguration(non_blocking=True)
        )
        if self.style_bank is not None:
            self.style_bank = self.style_bank.to(self.accelerator.device)
        if self.stain_bank is not None:
            self.stain_bank = self.stain_bank.float().to(self.accelerator.device)
        self.console = get_console()
        if self.channels_last:
            # NHWC lets cuDNN pick Tensor-Core kernels for bf16
            self.model = self.model.to(memory_format=torch.channels_last)

        # EMA of the weights (Arpit et al., NeurIPS 2022): validated, selected and saved instead of the raw weights
        self.ema = None
        if ema_decay > 0:
            from timm.utils import ModelEmaV3
            self.ema = ModelEmaV3(self.model, decay=float(ema_decay), device=self.accelerator.device)
        self._step = 0
        # after the EMA copy, which would otherwise clone the hooks
        self.mixstyle = None
        if mixstyle_p > 0:
            self.mixstyle = MixStyle(mixstyle_p, mixstyle_alpha).attach([self.model.stages[i] for i in mixstyle_stages])

        self.optimizer, self.scheduler = setup_optimizer(
            model=self.model,
            lr=lr,
            layer_decay=layer_decay,
            weight_decay=weight_decay,
            total_epochs=epochs - self.probe_epochs,
            warmup_epochs=warmup_epochs
        )

        self.criterion = WeightedCrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

        # the Azure-B cell prior behind every MorphoMix component (computed in the loader workers when possible)
        self.cam_extractor = None
        if self.augmentation in MORPHO_ARMS:
            from src.cam.prior_only import PriorOnlyCAM
            self.cam_extractor = PriorOnlyCAM()

        # The scheduler stays outside prepare(): AcceleratedScheduler skips its step whenever the last
        # optimizer step was skipped by the fp16 GradScaler, which shifts a per-epoch schedule.
        self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader)
        if len(self.train_loader) == 0:  # drop_last: a split below one batch would save an untrained model
            raise ValueError(f"{experiment_id}: train split is smaller than one batch; lower batch_size or --limit")

        self.ckpt_dir = os.path.join(output_dir, "checkpoints")
        self.cm_dir = os.path.join(output_dir, "figures", "confusion_matrices")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.cm_dir, exist_ok=True)
        self.ckpt_path = os.path.join(self.ckpt_dir, f"{self.experiment_id}_best.pt")

        self.csv_logger = UnifiedCSVLogger(output_dir=output_dir, filename="training_log.csv")
        self.resources = ResourceMonitor(os.path.join(output_dir, "logs"), experiment_id, "phase1", resources)

    def _augment(self, images: torch.Tensor, targets: torch.Tensor,
                 priors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The arm's augmentation of the normalised GPU batch; no arm changes the labels (`targets` only tell C4
        which cells share a class)."""
        if self.augmentation in MORPHO_ARMS:
            from src.augmentations.morpho_mix import apply_morpho_mix
            cam_maps = self.cam_extractor.generate(images, priors=priors)
            images, _ = apply_morpho_mix(
                images, cam_maps,
                threshold=self.morpho_threshold,
                bank=self.style_bank,
                appearance_prob=self.appearance_prob,
                appearance_alpha=self.appearance_alpha,
                virtual_template_prob=self.virtual_template_prob,
                acquisition_prob=self.acquisition_prob,
                small_cell_prob=self.small_cell_prob,
                small_cell_px=self.small_cell_px,
                small_cell_lowdetail_prob=self.small_cell_lowdetail_prob,
                field_prob=self.field_prob,
                field_cells=self.field_cells,
                labels=targets,
                use_background=self.use_background,
                background_prob=self.background_prob,
                background_rbc=self.background_rbc,
                field_rbc=self.field_rbc,
                feather_edges=self.feather_edges,
                rsn_prob=self.rsn_prob,
                rsn_stats=self.randstainna_stats,
                rsn_std_hyper=self.randstainna_std_hyper,
            )
        elif self.augmentation in STAIN_ARMS:
            from src.augmentations import stain_baselines as sb
            fg = sb.foreground(images)
            if self.augmentation == "hed_jitter":
                images = sb.apply_hed_jitter(images, fg, sigma=self.hed_sigma, prob=self.stain_prob)
            elif self.augmentation == "randstainna":
                images = sb.apply_randstainna(images, fg, self.randstainna_stats, std_hyper=self.randstainna_std_hyper,
                                              prob=self.stain_prob)
            else:
                images = sb.apply_stain_mixup(images, fg, self.stain_bank, prob=self.stain_prob)
        if self.lowres_prob > 0:  # last, on the arm's output: the capture of the finished image
            images = degrade_resolution(images, self.lowres_prob, self.lowres_scale)
        return images

    def _freeze(self, frozen: bool) -> None:
        """Stem and the first `freeze_stages` stages. Their parameters stay in the optimizer: a parameter without a
        gradient is skipped by AdamW (no step, no weight decay) and gets fresh moments when it is released."""
        model = self.accelerator.unwrap_model(self.model)
        for module in (model.stem, *model.stages[:self.freeze_stages]):
            module.requires_grad_(not frozen)

    OOM_SIGNATURES = ("out of memory", "device not ready", "failed to create gpu mapping",
                      "cuda error: unknown error")

    def _forward_guarded(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with an actionable message instead of a bare CUDA failure mid-grid."""
        try:
            return self.model(x)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            message = str(exc).lower()
            is_oom = isinstance(exc, torch.cuda.OutOfMemoryError)
            if not is_oom and not any(sig in message for sig in self.OOM_SIGNATURES):
                raise
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"GPU memory exhausted in {self.experiment_id} "
                f"(batch {x.shape[0]}, arm {self.augmentation}). "
                f"Lower `batch_size` in configs/config.yaml - the same value for every arm - then re-run the "
                f"grid. Original error: {exc}"
            ) from exc

    def train_epoch(
            self,
            progress: Optional[Progress] = None,
            task: Optional[TaskID] = None,
            shown: Optional[Dict[str, float]] = None,
    ) -> float:
        """One epoch; `shown` holds the metrics on the bar (train_loss is refreshed every `log_every` steps)."""
        shown = {} if shown is None else shown
        self.model.train()
        total_loss = torch.zeros((), device=self.accelerator.device)
        num_batches = len(self.train_loader)

        for step, batch in enumerate(self.resources.timed(self.train_loader, "train")):
            images, targets = batch[0], batch[1]
            priors = batch[3] if len(batch) > 3 else None  # Azure-B priors precomputed in the workers
            with self.accelerator.accumulate(self.model):
                images = self._augment(images, targets, priors)
                if self.channels_last:  # after the augmentation, whose outputs are plain contiguous tensors
                    images = images.contiguous(memory_format=torch.channels_last)
                outputs = self._forward_guarded(images)
                loss = self.criterion(outputs, targets)

                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.ema is not None and self.accelerator.sync_gradients:
                    self._step += 1
                    self.ema.update(self.accelerator.unwrap_model(self.model), step=self._step)

            total_loss += loss.detach()
            # .item() forces a device sync
            if progress is not None and task is not None:
                if (step + 1) % self.log_every == 0 or step + 1 == num_batches:
                    shown["train_loss"] = loss.item()
                    progress.update(task, advance=1, metrics=format_metrics(shown))
                else:
                    progress.update(task, advance=1)

        if not self.probing:
            self.scheduler.step()
        return total_loss.item() / max(1, num_batches)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, progress: Optional[Progress] = None,
                 model: Optional[nn.Module] = None) -> Dict[str, Any]:
        """Runs `model` (default: the trained model) over `loader`; compute_metrics() output plus val_loss."""
        model = self.model if model is None else model
        model.eval()
        total_loss = 0.0
        y_true, y_pred, probs, names = [], [], [], []
        num_batches = len(loader)
        task = progress.add_task(VALIDATION_DESCRIPTION, total=num_batches, metrics="") if progress else None

        for batch in self.resources.timed(loader, "val"):
            images, targets = batch[0], batch[1]
            images = images.to(self.accelerator.device, non_blocking=True)
            if self.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            targets = targets.to(self.accelerator.device, non_blocking=True)
            # never label-smoothed, so val losses compare across recipes; reduction="none" + mean as in training
            weight = self.criterion.weight
            weight = weight.to(images.device) if weight is not None else None
            # the EMA copy is not wrapped by accelerate, so autocast is set here for both
            with self.accelerator.autocast():
                if self.tta_views > 1:
                    # tta_logits returns log-probabilities of the view-averaged distribution
                    log_probs = tta_logits(model, images, n_views=self.tta_views).float()
                    loss = F.nll_loss(log_probs, targets, weight=weight, reduction="none").mean()
                    batch_probs = log_probs.exp()
                else:
                    outputs = model(images).float()
                    loss = F.cross_entropy(outputs, targets, weight=weight, reduction="none").mean()
                    batch_probs = torch.softmax(outputs, dim=-1)
            total_loss += loss.item()

            probs.extend(batch_probs.float().cpu().numpy().tolist())
            y_true.extend(targets.cpu().numpy().tolist())
            # in-training val rule: ALL iff p(ALL) >= 0.5 (argmax would send a tie to Normal); the test reports use
            # inference.decision_threshold
            pred = ((batch_probs[:, 1] >= DECISION_THRESHOLD).long() if self.num_classes == 2
                    else batch_probs.argmax(dim=-1))
            y_pred.extend(pred.cpu().numpy().tolist())
            names.extend(batch[2])
            if task is not None:
                progress.update(task, advance=1)

        if task is not None:
            progress.remove_task(task)
        metrics = compute_metrics(y_true, y_pred, self.class_names)
        if self.num_classes == 2:
            metrics["roc_auc"] = threshold_free_metrics(y_true, [p[1] for p in probs])["roc_auc"]
        metrics["val_loss"] = total_loss / max(1, num_batches)
        metrics["y_true"] = y_true
        metrics["y_pred"] = y_pred
        metrics["probs"] = probs
        metrics["names"] = names
        return metrics

    def _selected_model(self) -> nn.Module:
        """The weights that are validated and saved: the EMA copy when enabled."""
        return self.ema.module if self.ema is not None else self.accelerator.unwrap_model(self.model)

    def validate(self, progress: Optional[Progress] = None) -> Dict[str, Any]:
        return self.evaluate(self.val_loader, progress, model=self._selected_model())

    def load_best(self) -> None:
        """Reloads the best checkpoint weights (EMA ones when enabled) into the trained model for test scoring."""
        state = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        self.accelerator.unwrap_model(self.model).load_state_dict(state["model_state_dict"])

    def run(self) -> Dict[str, Any]:
        best_score = -1.0
        best_val_loss = float("inf")
        best_epoch = 0
        best_metrics: Dict[str, Any] = {}
        curve: List[Dict[str, float]] = []
        if self.freeze_stages:
            self._freeze(True)

        # Lightning's RichProgressBar: one bar reset every epoch, so earlier epochs leave no lines behind
        shown: Dict[str, float] = {}
        task = None
        with create_progress_bar() as progress, self.resources:
            for epoch in range(self.epochs):
                if self.freeze_stages and self.freeze_epochs and epoch == self.freeze_epochs:
                    self._freeze(False)
                if self.probe_epochs and epoch <= self.probe_epochs:
                    self.probing = epoch < self.probe_epochs
                    n = len(self.optimizer.param_groups)
                    lrs = [self.probe_lr] * n if self.probing else self.scheduler.get_last_lr()
                    for group, lr in zip(self.optimizer.param_groups, lrs):
                        group["lr"] = lr
                self.resources.begin_epoch(epoch + 1)
                current_lr = self.optimizer.param_groups[-1]["lr"]  # the lr this epoch trains at
                description = epoch_description(epoch, self.epochs)
                if task is None:
                    task = progress.add_task(description, total=len(self.train_loader), metrics="")
                else:
                    progress.reset(task, total=len(self.train_loader), description=description, visible=True)
                train_loss = self.train_epoch(progress, task, shown)
                val_results = self.validate(progress)
                usage = self.resources.end_epoch()

                val_loss = val_results["val_loss"]
                val_f1 = val_results["macro_f1"]
                val_acc = val_results["accuracy"]
                curve.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                              "val_roc_auc": val_results.get("roc_auc", float("nan"))})
                shown.update(train_loss=train_loss, val_loss=val_loss, val_f1=val_f1)
                progress.update(task, metrics=format_metrics(shown))
                if not self.console.is_interactive:  # no live bar (Colab cell, log file): one line per epoch
                    self.console.print(f"{self.experiment_id} epoch {epoch + 1}/{self.epochs} | lr {current_lr:.2e} | "
                                       f"train loss {train_loss:.4f} | val loss {val_loss:.4f} | val AUC "
                                       f"{val_results.get('roc_auc', float('nan')):.4f} | val F1 {val_f1:.4f}{usage}")

                # select on `selection_metric`, ties broken by val loss; NaN (one-class val) never wins
                score = val_results[self.selection_metric]
                is_best = (score > best_score) or (score == best_score and val_loss < best_val_loss)
                if is_best:
                    best_score = score
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    best_metrics = val_results

                    if self.accelerator.is_main_process:
                        unwrapped = self._selected_model()
                        torch.save({
                            "epoch": epoch + 1,
                            "experiment_id": self.experiment_id,
                            "augmentation": self.augmentation,
                            "seed": self.seed,
                            "deterministic": torch.are_deterministic_algorithms_enabled(),
                            "ema": self.ema is not None,
                            "class_names": self.class_names,
                            "model_state_dict": unwrapped.state_dict(),
                            "val_loss": val_loss,
                            "macro_f1": val_f1,
                            "metrics": {k: v for k, v in val_results.items() if k not in ("probs", "names")}
                        }, self.ckpt_path)

                if self.accelerator.is_main_process:
                    self.csv_logger.log_epoch(
                        experiment_id=self.experiment_id,
                        model=self.model_name,
                        augmentation=self.augmentation,
                        seed=self.seed,
                        epoch=epoch + 1,
                        total_epochs=self.epochs,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        val_macro_f1=val_f1,
                        val_roc_auc=val_results.get("roc_auc", float("nan")),
                        val_balanced_acc=val_results.get("balanced_accuracy", 0.0),
                        val_accuracy=val_acc,
                        learning_rate=current_lr,
                        is_best=is_best,
                        per_class_f1_dict=val_results.get("per_class_f1", {})
                    )

        if self.resources.rows:
            self.console.print(f"[dim]{self.resources.run_line()}[/dim]")
        if self.cam_extractor is not None:
            self.cam_extractor.remove_hooks()
        if self.mixstyle is not None:
            self.mixstyle.remove()
        if best_epoch == 0:  # every val score was NaN (diverged run or one-class val)
            raise RuntimeError(f"{self.experiment_id}: no epoch was ever selected (val {self.selection_metric} never "
                               f"finite); {self.ckpt_path} is absent or stale, so the run is not scored")

        # validation matrix of the selected epoch
        if best_metrics and "confusion_matrix" in best_metrics:
            save_confusion(
                cm=np.array(best_metrics["confusion_matrix"]),
                class_names=self.class_names,
                save_path=os.path.join(self.cm_dir, f"{self.experiment_id}_val_cm.png"),
                heading="LeukemiaAttri val",
            )

        return {
            "experiment_id": self.experiment_id,
            "model": self.model_name,
            "augmentation": self.augmentation,
            "seed": self.seed,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "selection_metric": self.selection_metric,
            "macro_f1": best_metrics.get("macro_f1", 0.0),
            "roc_auc": best_metrics.get("roc_auc", float("nan")),
            "balanced_accuracy": best_metrics.get("balanced_accuracy", 0.0),
            "accuracy": best_metrics.get("accuracy", 0.0),
            "macro_precision": best_metrics.get("macro_precision", 0.0),
            "macro_recall": best_metrics.get("macro_recall", 0.0),
            "cohen_kappa": best_metrics.get("cohen_kappa", 0.0),
            "macro_specificity": best_metrics.get("macro_specificity", 0.0),
            "per_class_f1": best_metrics.get("per_class_f1", {}),
            "curve": curve,
            "checkpoint": self.ckpt_path
        }
