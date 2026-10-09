"""Config loading and repo-root resolution shared by every script."""
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """`path` defaults to $MORPHOMIX_CONFIG, else configs/config.yaml; every script and stage reads the same file,
    so a smoke test of `all` points MORPHOMIX_CONFIG at a copy with its own results_dir."""
    path = path or os.environ.get("MORPHOMIX_CONFIG", "configs/config.yaml")
    with open(ROOT / path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """`{key: value}` overrides in place; a dotted key (`phase1.base_lr`) sets a nested value. Every key must already
    exist in the config, so a typo (`apperance_prob`) raises KeyError instead of passing as a no-op."""
    for key, value in overrides.items():
        *parents, leaf = key.split(".")
        node = cfg
        for name in parents:
            if not isinstance(node.get(name), dict):
                raise KeyError(f"override {key!r}: {name!r} is not a config section")
            node = node[name]
        if leaf not in node:
            raise KeyError(f"override {key!r}: {leaf!r} is not a config key")
        node[leaf] = value
    return cfg


def refuse_real_results_dir(results_dir: str) -> None:
    """A smoke run (`--limit`) must never leave truncated runs where the real plan resumes from: refuse the
    results_dir of configs/config.yaml, compared as resolved paths."""
    if same_path(results_dir, load_config("configs/config.yaml")["results_dir"]):
        raise SystemExit("--limit would leave smoke runs where the real plan resumes from; use --set "
                         "results_dir=agent_space/smoke, or point MORPHOMIX_CONFIG at a config copy with its own "
                         "results_dir")


def abs_path(rel: str) -> str:
    """Resolve a repo-relative path to an absolute one (no-op for absolute inputs)."""
    return rel if os.path.isabs(rel) else str(ROOT / rel)


def same_path(a: str, b: str) -> bool:
    """Whether two repo-relative or absolute paths name one location (`results`, `./results/`, a symlink to it)."""
    return len({os.path.normcase(os.path.realpath(abs_path(p))) for p in (a, b)}) == 1


def setup_cuda_env() -> None:
    """Fragmentation-safe allocator and the cuBLAS workspace that deterministic mode needs."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
