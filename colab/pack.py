"""Build the Colab archives (run from anywhere): morphomix_code.zip and morphomix_data.zip (cohorts, splits, the MLL23
style bank, the RandStainNA fit and, with aux_train, the Bodzas auxiliary cells and their colour-matched copy; holds
the non-redistributable ALL-IDB2 images: private Drive only). No MLL23 image is shipped: the bank holds statistics
only."""
import argparse
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.utils.config import load_config  # noqa: E402

CODE_SKIP = {".venv", "data", "agent_space", ".claude", ".agents", ".git", "paper", "kampus", "Bimbingan",
             "__pycache__", ".ruff_cache"}
COHORTS = ("train", "val", "test_allidb2", "test_aria")  # scripts/train.py: TRAIN_DIR / VAL_DIR / TEST_SETS
STORED = {".png", ".jpg", ".jpeg"}  # already compressed


def pack(archive: Path, files) -> None:
    n = 0
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(ROOT).as_posix(),
                    compress_type=zipfile.ZIP_STORED if f.suffix.lower() in STORED else zipfile.ZIP_DEFLATED)
            n += 1
    print(f"{archive}: {n} files, {archive.stat().st_size / 1e6:.1f} MB")


def data_paths(cfg: dict) -> list:
    """What `prepare` and `split` build, from the config; data/raw stays local."""
    data_dir = cfg["phase1"]["data_dir"]
    bank = cfg["style_bank"]["path"]
    splits = {os.path.dirname(cfg[k]["csv"]) for k in ("aria_groups", "allidb2_groups")}
    built = [f"{data_dir}/cohorts.json", *sorted(splits), bank, os.path.splitext(bank)[0] + ".json"]
    aux = [cfg["aux_train"]["dir"], cfg["aux_train"]["colour_matched_dir"]] if cfg["aux_train"]["enabled"] else []
    return [f"{data_dir}/{c}" for c in COHORTS] + built + [cfg["randstainna_stats"]] + aux


def tree(paths) -> list:
    missing = [d for d in paths if not (ROOT / d).exists()]
    if missing:
        raise SystemExit(f"missing {missing}; run `python main.py prepare` and `python main.py split` first")
    return sorted(f for d in paths for f in ([ROOT / d] if (ROOT / d).is_file() else (ROOT / d).rglob("*"))
                  if f.is_file() and f.suffix != ".tmp")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(ROOT), help="directory for the archives")
    p.add_argument("--skip", nargs="+", default=[], choices=["code", "data"], help="archives not rebuilt")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if "code" not in a.skip:
        code = [f for f in ROOT.rglob("*") if f.is_file() and f.suffix not in (".zip", ".pyc")
                and not CODE_SKIP.intersection(f.relative_to(ROOT).parts)
                and not f.relative_to(ROOT).parts[0].startswith("results")]  # every results folder
        pack(out / "morphomix_code.zip", sorted(code))
    if "data" not in a.skip:
        pack(out / "morphomix_data.zip", tree(data_paths(load_config())))


if __name__ == "__main__":
    main()
