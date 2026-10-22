"""DataLoader factory using the hardware settings in the config."""
from typing import Dict, Optional

import cv2
import torch
from torch.utils.data import DataLoader, Dataset


def _single_threaded_opencv(_worker_id: int) -> None:
    """The Azure-B prior runs OpenCV in every worker; one thread each avoids oversubscribing the vCPUs."""
    cv2.setNumThreads(1)


def make_loader(
    ds: Dataset,
    cfg: Dict,
    shuffle: bool,
    seed: Optional[int] = None,
    drop_last: bool = False,
    workers: Optional[int] = None,
) -> DataLoader:
    """`workers` overrides cfg['num_workers'] (eval loaders pass cfg['eval_num_workers'])."""
    nw = int(cfg.get("num_workers", 0)) if workers is None else int(workers)
    kwargs = dict(
        batch_size=cfg["batch_size"],
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=True,
        drop_last=drop_last,
    )
    if nw > 0:
        # also in the parent, which may have used OpenCV already: a forked worker inherits no running thread pool
        cv2.setNumThreads(1)
        kwargs["persistent_workers"] = bool(cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))
        kwargs["worker_init_fn"] = _single_threaded_opencv
    if shuffle and seed is not None:
        kwargs["generator"] = torch.Generator().manual_seed(seed)
    return DataLoader(ds, **kwargs)


def eval_loader(ds: Dataset, cfg: Dict) -> DataLoader:
    """Deterministic, non-shuffled loader for val/test with cfg['eval_num_workers'] workers."""
    return make_loader(ds, cfg, shuffle=False, workers=cfg.get("eval_num_workers", 0))
