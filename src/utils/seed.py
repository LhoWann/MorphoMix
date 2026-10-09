import os
import random
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seeds Python, NumPy and torch (CPU + CUDA)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_determinism(cfg: Dict) -> bool:
    """Turn on bit-reproducible kernels when `deterministic` is set, and say so."""
    want = bool(cfg.get("deterministic", False))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = not want
    torch.backends.cudnn.deterministic = want
    try:
        torch.use_deterministic_algorithms(want)
    except Exception as exc:  # an op without a deterministic kernel
        print(f"  deterministic mode unavailable: {exc}")
        torch.use_deterministic_algorithms(False)
        return False
    return want


def setup_run(cfg: Dict, seed: int) -> None:
    """Seed, determinism and matmul precision; called by every training run."""
    apply_determinism(cfg)
    torch.set_float32_matmul_precision("high")
    set_seed(seed)
