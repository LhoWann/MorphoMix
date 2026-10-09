"""Data preparation (`prepare`): data/raw -> data/processed/{train, val, test_allidb2, test_aria}, one laboratory
each, plus the stain references of the augmentation arms (MLL23 style bank, RandStainNA training-set statistics).
Test-set groups for the paired bootstrap (`split`, `split --allidb2`)."""
import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import v2

from src.datasets.dataset import mask_annotation
from src.datasets.leukemiaattri import build_val_cohort
from src.datasets.sampler import overlap_groups
from src.utils.config import load_config, abs_path, ROOT

RAW = ROOT / "data" / "raw"
CNMC = RAW / "C-NMC_2019"
CNMC_FOLDS = ("fold_0", "fold_1", "fold_2")
ALLIDB2 = RAW / "ALL-IDB2" / "img"
ARIA = RAW / "Aria_B-ALL" / "Original"
LEUKEMIAATTRI = RAW / "LeukemiaAttri" / "H_100X_C1"
CLASSES = ("Normal", "ALL")
CNMC_CLASSES = {"hem": "Normal", "all": "ALL"}
# Aria is a binary test cohort: Benign (hematogones) is Normal, every B-ALL subtype is ALL
ARIA_CLASSES = {"Benign": "Normal", "Early": "ALL", "Pre": "ALL", "Pro": "ALL"}
BODZAS = RAW / "Bodzas2023"
AUX_VERSION = 2  # of prepare_aux_cells; 2: cells cut by the crop border are skipped


def _square_image(src: Path) -> Image.Image:
    """RGB image, padded to a square by edge replication so the resize never changes the aspect ratio."""
    arr = np.asarray(Image.open(src).convert("RGB"))
    h, w = arr.shape[:2]
    if h == w:
        return Image.fromarray(arr)
    dy, dx = max(h, w) - h, max(h, w) - w
    return Image.fromarray(np.pad(arr, ((dy // 2, dy - dy // 2), (dx // 2, dx - dx // 2), (0, 0)), mode="edge"))


def _resize_save(src: Path, dst: Path, size: int, box: Optional[Sequence[float]] = None) -> None:
    """`src` squared, `box` corner painted out (Aria's scale bar) and resized to `size`, saved as PNG."""
    im = _square_image(src)
    if box is not None:
        im = mask_annotation(im, box)
    if im.size != (size, size):
        t = v2.functional.resize(v2.functional.to_image(im), [size, size], antialias=True)
        im = Image.fromarray(t.permute(1, 2, 0).numpy())
    # via a temp file: sync_dir skips existing files, so an interrupted write must not leave a truncated PNG
    tmp = dst.with_suffix(".tmp")
    im.save(tmp, format="PNG", compress_level=9)
    tmp.replace(dst)


def sync_dir(dest: Path, files: List[Tuple[Path, str]], size: int, box: Optional[Sequence[float]] = None,
             threads: int = 8) -> Tuple[int, int]:
    """Make `dest` contain exactly the given files as `<stem>.png` at `size`."""
    dest.mkdir(parents=True, exist_ok=True)
    wanted = {Path(name).stem + ".png": src for src, name in files}
    removed = 0
    for existing in dest.iterdir():
        if existing.name not in wanted:
            existing.unlink()
            removed += 1
    todo = [(src, dest / name) for name, src in wanted.items() if not (dest / name).exists()]
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(lambda t: _resize_save(t[0], t[1], size, box), todo))
    return len(todo), removed


def cohort_files() -> Dict[str, Dict[str, List[Tuple[Path, str]]]]:
    """{cohort: {class: [(source path, output name)]}} for the image-copy cohorts (val is cropped separately)."""
    cohorts: Dict[str, Dict[str, List[Tuple[Path, str]]]] = {c: {cls: [] for cls in CLASSES}
                                                             for c in ("train", "test_allidb2", "test_aria")}
    for fold in CNMC_FOLDS:
        for src_cls, cls in CNMC_CLASSES.items():
            bmps = sorted((CNMC / fold / src_cls).glob("*.bmp"))
            cohorts["train"][cls] += [(f, f"CNMC_{fold}_{f.name}") for f in bmps]
    for f in sorted(ALLIDB2.glob("*.tif")):
        cohorts["test_allidb2"]["ALL" if f.stem.endswith("_1") else "Normal"].append((f, f"ALLIDB_{f.name}"))
    for sub, cls in ARIA_CLASSES.items():
        cohorts["test_aria"][cls] += [(f, f"ARIA_{sub}_{f.name}") for f in sorted((ARIA / sub).glob("*.jpg"))]
    for name, by_class in cohorts.items():
        if not all(by_class.values()):
            raise FileNotFoundError(f"{name}: no images for {[c for c, v in by_class.items() if not v]} under {RAW}")
    return cohorts


def prepare_stain_references(cfg: Dict, train_changed: bool, n_train: int) -> None:
    """The MLL23 style bank (C1, stain_mixup), rebuilt only when missing or built from other inputs, and the
    RandStainNA Lab statistics of the training images, refitted when missing, when `train` changed or when the fit
    covers another number of images than the `n_train` now in train (an interrupted `prepare`)."""
    from src.augmentations.stain_baselines import fit_randstainna
    from src.augmentations.style_bank import bank_provenance, build_style_bank, provenance_path
    path = Path(abs_path(cfg["style_bank"]["path"]))
    wanted = json.loads(json.dumps(bank_provenance(cfg)))  # tuples as lists, as the sidecar holds them
    built = None
    if path.exists() and Path(provenance_path(str(path))).exists():
        with open(provenance_path(str(path)), encoding="utf-8") as f:
            built = json.load(f)
    if built is None or {k: built.get(k) for k in wanted} != wanted:
        start = time.time()
        built = build_style_bank(cfg)
        print(f"style bank: {built['n_valid']:,} of {built['n_images']:,} MLL23 cells ({time.time() - start:.0f} s, "
              f"{path.stat().st_size / 1e6:.1f} MB) -> {path}")
    else:
        print(f"style bank: {built['n_valid']:,} MLL23 cells, up to date ({path})")
    stats = Path(abs_path(cfg["randstainna_stats"]))
    fitted_n = None
    if stats.exists():
        with open(stats, encoding="utf-8") as f:
            fitted_n = json.load(f).get("n_images")
    if train_changed or fitted_n != n_train:
        fitted = fit_randstainna(str(Path(abs_path(cfg["phase1"]["data_dir"])) / "train"), img_size=cfg["img_size"])
        stats.parent.mkdir(parents=True, exist_ok=True)
        with open(str(stats) + ".tmp", "w", encoding="utf-8") as f:
            json.dump(fitted, f, indent=2)
        os.replace(str(stats) + ".tmp", stats)
        print(f"RandStainNA statistics of the training images -> {stats}")


def prepare_binary_datasets() -> None:
    """train = C-NMC (every fold), val = LeukemiaAttri cell crops, tests = ALL-IDB2 and Aria (binary), at img_size."""
    cfg = load_config()
    size, out = cfg["img_size"], Path(abs_path(cfg["phase1"]["data_dir"]))
    settings = {"img_size": size, "aria_annotation_box": list(cfg["aria_annotation_box"])}
    if (out / "cohorts.json").exists():  # sync_dir keeps existing PNGs, so other settings would leave them stale
        with open(out / "cohorts.json", encoding="utf-8") as f:
            built = json.load(f)
        stale = [k for k, v in settings.items() if k in built and built[k] != v]
        if stale:
            was, now = {k: built[k] for k in stale}, {k: settings[k] for k in stale}
            raise SystemExit(f"{out} was built with {was}, the config now has {now}; delete {out} and run "
                             f"`python main.py prepare` and `python main.py split` again")
    counts: Dict[str, Dict[str, int]] = {}
    train_changed = False
    for name, by_class in cohort_files().items():
        box = settings["aria_annotation_box"] if name == "test_aria" else None
        for cls, files in by_class.items():
            added, removed = sync_dir(out / name / cls, files, size, box)
            if added or removed:
                print(f"{name}/{cls}: +{added} -{removed}")
                train_changed |= name == "train"
        counts[name] = {cls: len(files) for cls, files in by_class.items()}
    manifest = build_val_cohort(LEUKEMIAATTRI, out / "val", out_size=size)
    counts["val"] = manifest["class_name"].value_counts().reindex(list(CLASSES), fill_value=0).astype(int).to_dict()

    sources = {"train": "C-NMC 2019, folds 0-2 (SBILab, AIIMS New Delhi)",
               "val": "LeukemiaAttri H_100X_C1 lymphoblast / lymphocyte crops (Chughtai Labs, Lahore)",
               "test_allidb2": "ALL-IDB2 (Universita degli Studi di Milano)",
               "test_aria": "Aria et al. full fields, Benign = Normal, Early/Pre/Pro = ALL (Taleqani Hospital, Tehran)"}
    with open(out / "cohorts.json", "w", encoding="utf-8") as f:
        json.dump({**settings, "counts": counts, "sources": sources}, f, indent=2)
    for name in ("train", "val", "test_allidb2", "test_aria"):
        c = counts[name]
        print(f"{name:<13} {c['Normal'] + c['ALL']:>6,} (Normal {c['Normal']:,} / ALL {c['ALL']:,})  {sources[name]}")
    prepare_stain_references(cfg, train_changed, sum(counts["train"].values()))
    if cfg["aux_train"]["enabled"]:
        prepare_aux_cells(cfg)
        prepare_colour_matched_aux(cfg)


def prepare_aux_cells(cfg: Dict) -> None:
    """`aux_train`: Bodzas et al. 2023 cells in the C-NMC format (`smear_cell`, the centre cell cut onto black) under
    `dir/{Normal,ALL}`: `n_positive` / `n_negative` per label drawn in a seed-0 permutation of the sorted files of
    its classes; a cell whose mask is empty, touches the frame or covers more than `max_area` of it (a cluster, a
    failed mask) is skipped. Rebuilt only when its settings change."""
    from src.datasets.cells import list_images, load_rgb, smear_cell
    aux = cfg["aux_train"]
    out = Path(abs_path(aux["dir"]))
    settings = {k: aux[k] for k in ("positive", "negative", "n_positive", "n_negative", "max_area")}
    settings.update(img_size=cfg["img_size"], version=AUX_VERSION)
    if (out / "aux.json").exists():
        with open(out / "aux.json", encoding="utf-8") as f:
            if json.load(f)["settings"] == json.loads(json.dumps(settings)):
                print(f"aux_train: up to date ({out})")
                return
    start, counts = time.time(), {}
    (out / "aux.json").unlink(missing_ok=True)  # absent while the folder is rebuilt, so nothing trains on a partial set
    labels = (("ALL", aux["positive"], aux["n_positive"]), ("Normal", aux["negative"], aux["n_negative"]))
    for cls, sources, n in labels:
        dest = out / cls
        dest.mkdir(parents=True, exist_ok=True)
        for old in dest.glob("*.png"):
            old.unlink()
        files = sorted(f for c in sources for f in list_images(str(BODZAS / c)))
        kept = 0
        for i in np.random.default_rng(0).permutation(len(files)):
            rgb = load_rgb(files[i], cfg["img_size"])
            m = smear_cell(rgb) > 0
            # smear_cell keeps a 3 px background frame, so a cell cut by the crop border ends 3 px inside it
            edge = m[:4].any() or m[-4:].any() or m[:, :4].any() or m[:, -4:].any()
            if not m.any() or m.mean() > aux["max_area"] or edge:
                continue
            name = f"bodzas2023_{Path(files[i]).parent.name}_{Path(files[i]).stem}.png".replace(" ", "_")
            Image.fromarray(rgb * m[..., None].astype(np.uint8)).save(dest / name)
            kept += 1
            if kept == n:
                break
        counts[cls] = kept
    digest = hashlib.sha256()
    for path in sorted(out.glob("*/*.png")):
        digest.update(f"{path.parent.name}/{path.name}".encode())
        digest.update(path.read_bytes())
    with open(out / "aux.json.tmp", "w", encoding="utf-8") as f:  # train.used_references hashes this file
        json.dump({"settings": settings, "counts": counts, "images_sha256": digest.hexdigest()}, f, indent=2)
    os.replace(out / "aux.json.tmp", out / "aux.json")
    print(f"aux_train: Bodzas 2023 ALL {counts['ALL']:,} / Normal {counts['Normal']:,} "
          f"({time.time() - start:.0f} s), {out}")


def prepare_colour_matched_aux(cfg: Dict) -> None:
    """The same auxiliary cells with their cell pixels Reinhard-matched in Lab to the mean C-NMC template
    (`randstainna_stats` mean_avg / sd_avg) under `aux_train.colour_matched_dir`: every Bodzas cell gets one colour,
    so the slide colour that separates its lymphoblasts from its normal cells is gone (control arm g). Rebuilt
    when the source set or the fit changes."""
    import torch
    from src.augmentations.background import CELL_LUMINANCE_FLOOR
    from src.augmentations.stain import SD_FLOOR, lab_to_rgb, region_moments, rgb_to_lab
    aux = cfg["aux_train"]
    src, out = Path(abs_path(aux["dir"])), Path(abs_path(aux["colour_matched_dir"]))
    with open(src / "aux.json", encoding="utf-8") as f:
        source = json.load(f)
    with open(abs_path(cfg["randstainna_stats"]), encoding="utf-8") as f:
        stats = json.load(f)
    settings = {"source_sha256": source["images_sha256"], "mean": stats["mean_avg"], "sd": stats["sd_avg"]}
    if (out / "aux.json").exists():
        with open(out / "aux.json", encoding="utf-8") as f:
            if json.load(f)["settings"] == settings:
                print(f"aux_train colour-matched: up to date ({out})")
                return
    (out / "aux.json").unlink(missing_ok=True)
    target_mean = torch.tensor(stats["mean_avg"], dtype=torch.float64).view(1, 3, 1, 1)
    target_sd = torch.tensor(stats["sd_avg"], dtype=torch.float64).view(1, 3, 1, 1)
    digest = hashlib.sha256()
    for cls in source["counts"]:
        dest = out / cls
        dest.mkdir(parents=True, exist_ok=True)
        for old in dest.glob("*.png"):
            old.unlink()
        for path in sorted((src / cls).glob("*.png")):
            rgb = torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1)[None].double() / 255
            fg = rgb.mean(dim=1, keepdim=True) >= CELL_LUMINANCE_FLOOR
            lab = rgb_to_lab(rgb)
            mean, sd, _ = region_moments(lab, (fg,))
            mean, sd = mean[:, 0, :, None, None], sd[:, 0, :, None, None].clamp(min=SD_FLOOR)
            matched = torch.where(fg, lab_to_rgb((lab - mean) / sd * target_sd + target_mean).clamp(0, 1), rgb)
            img = (matched[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            Image.fromarray(img).save(dest / path.name)
            digest.update(f"{cls}/{path.name}".encode())
            digest.update((dest / path.name).read_bytes())
    with open(out / "aux.json.tmp", "w", encoding="utf-8") as f:
        json.dump({"settings": settings, "counts": source["counts"], "images_sha256": digest.hexdigest()}, f,
                  indent=2)
    os.replace(out / "aux.json.tmp", out / "aux.json")
    print(f"aux_train colour-matched to the C-NMC template -> {out}")


# --- `split`: test-set groups ---

SPLIT_DOC = """Aria field groups: repeated captures of one smear field share a group, which the paired bootstrap of
`main.py stats` resamples as a unit (the captures are not independent test images). `--allidb2` does the same for
ALL-IDB2 crops of one cell."""
SUBTYPES = list(ARIA_CLASSES)  # Benign, Early, Pre, Pro; groups only link images of one subtype


def processed_name(subtype: str, raw_stem: str) -> str:
    """File name `prepare` gives an Aria image under data/processed/test_aria."""
    return f"ARIA_{subtype}_{raw_stem}.png"


def _canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Rows sorted by file, group ids renumbered 0..G-1 in order of first appearance, so that equal partitions
    give identical files whichever way they were computed."""
    df = df.sort_values("file").reset_index(drop=True)
    first = {g: i for i, g in enumerate(dict.fromkeys(df["group"]))}
    df["group"] = df["group"].map(first).astype(int)
    df["label"] = [CLASSES.index(ARIA_CLASSES.get(c, c)) for c in df["class_name"]]  # Aria subtype or Normal / ALL
    return df[["file", "class_name", "label", "group"]]


def groups_from_csv(path: str) -> pd.DataFrame:
    """Reuse the groups of an earlier table with `class_name`, `relative_path` (raw image) and `group` columns."""
    old = pd.read_csv(abs_path(path))
    stems = [os.path.splitext(os.path.basename(p))[0] for p in old["relative_path"]]
    return _canonical(pd.DataFrame({"file": [processed_name(c, s) for c, s in zip(old["class_name"], stems)],
                                    "class_name": old["class_name"], "group": old["group"]}))


def recompute_groups(cfg: Dict, threads: int) -> pd.DataFrame:
    """SIFT + RANSAC + overlap NCC over every same-subtype pair of raw Aria images (`overlap_groups`)."""
    spec = cfg["aria_groups"]
    files = [(sub, f) for sub in SUBTYPES for f in sorted((ARIA / sub).glob("*.jpg"))]
    if not files:
        raise FileNotFoundError(f"No Aria images under {ARIA}")
    groups = overlap_groups([str(f) for _, f in files], [SUBTYPES.index(sub) for sub, _ in files],
                            int(spec["min_inliers"]), float(spec["min_ncc"]), float(spec["min_overlap"]),
                            cfg["aria_annotation_box"], threads)
    return _canonical(pd.DataFrame({"file": [processed_name(sub, f.stem) for sub, f in files],
                                    "class_name": [sub for sub, _ in files], "group": groups}))


def allidb2_groups(cfg: Dict, threads: int) -> pd.DataFrame:
    """Processed ALL-IDB2 crops of one cell (cut from overlapping ALL-IDB1 fields): the Aria matching over every
    pair, across classes too, so that a same-cell pair with two labels would show as a mixed group."""
    spec = cfg["aria_groups"]
    root = os.path.join(abs_path(cfg["phase1"]["data_dir"]), "test_allidb2")
    files = sorted((f, cls) for cls in CLASSES for f in os.listdir(os.path.join(root, cls)) if f.endswith(".png"))
    if not files:
        raise FileNotFoundError(f"No ALL-IDB2 images under {root}; run `python main.py prepare`.")
    groups = overlap_groups([os.path.join(root, cls, f) for f, cls in files], [0] * len(files),
                            int(spec["min_inliers"]), float(spec["min_ncc"]), float(spec["min_overlap"]), None, threads)
    return _canonical(pd.DataFrame({"file": [f for f, _ in files], "class_name": [c for _, c in files],
                                    "group": groups}))


def check_processed(df: pd.DataFrame, cfg: Dict) -> None:
    """The group table must name exactly the images `prepare` wrote, or the bootstrap cannot find them."""
    root = os.path.join(abs_path(cfg["phase1"]["data_dir"]), "test_aria")
    if not os.path.isdir(root):
        return
    on_disk = {f for cls in CLASSES if os.path.isdir(os.path.join(root, cls))
               for f in os.listdir(os.path.join(root, cls))}
    if on_disk != set(df["file"]):
        missing, extra = sorted(on_disk - set(df["file"])), sorted(set(df["file"]) - on_disk)
        raise RuntimeError(f"Aria groups do not match {root}: {len(missing)} images without a group "
                           f"(e.g. {missing[:2]}), {len(extra)} grouped images not on disk (e.g. {extra[:2]})")


def split_main(argv: Optional[list] = None) -> int:
    cfg = load_config()
    p = argparse.ArgumentParser(description=SPLIT_DOC, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from-csv", metavar="PATH", default=None,
                   help="reuse the groups of an earlier table (columns class_name, relative_path, group) "
                        "instead of recomputing them from data/raw")
    p.add_argument("--threads", type=int, default=0, help="matching threads (default: every core)")
    p.add_argument("--allidb2", action="store_true",
                   help="group the processed ALL-IDB2 crops of one cell instead (-> allidb2_groups.csv)")
    a = p.parse_args(argv)

    if a.allidb2:
        df, out, name = allidb2_groups(cfg, a.threads), abs_path(cfg["allidb2_groups"]["csv"]), "ALL-IDB2"
        mixed = df.groupby("group")["label"].nunique() > 1
        if mixed.any():
            print(f"[WARN] {int(mixed.sum())} same-cell groups hold both labels: {list(mixed[mixed].index)}")
    else:
        df = groups_from_csv(a.from_csv) if a.from_csv else recompute_groups(cfg, a.threads)
        check_processed(df, cfg)
        out, name = abs_path(cfg["aria_groups"]["csv"]), "Aria"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    sizes = df["group"].value_counts()
    print(f"[OK] {len(df)} {name} images in {len(sizes)} groups ({int((sizes > 1).sum())} with repeats, "
          f"{int(sizes[sizes > 1].sum())} images; largest {int(sizes.max())}) -> {out}")
    for cls in CLASSES:
        sub = df[df["label"] == CLASSES.index(cls)]
        print(f"[OK] {cls:<6} {len(sub):5d} images, {sub['group'].nunique():5d} groups")
    return 0
