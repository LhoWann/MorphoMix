"""LeukemiaAttri (Chughtai Labs, CC BY-NC-SA 4.0) single-cell crops: the cross-lab validation cohort.

Full 640x640 fields at 100x with COCO boxes; only lymphoblast (ALL) and lymphocyte (Normal) boxes are used, from
both of the dataset's own splits, since none of it is trained on. Crops keep the real smear background.
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import v2

from src.utils.config import abs_path

CATEGORY_LABELS = {"lymphocyte": 0, "lymphoblast": 1}
CLASS_NAMES = ("Normal", "ALL")
IMAGE_RE = re.compile(r"^(?P<slide>\d+)_(?P<image>\d+)_(?P<mag>\d+)(?:_(?P<type>[A-Za-z]+))?$")
MIN_BOX_SIDE = 8          # px; smaller boxes are annotation slivers
DUPLICATE_IOU = 0.5       # two same-class boxes on one field overlapping this much are one cell
EDGE_MARGIN = 3           # px; a box this close to the field edge holds a cell cut by it (checked by eye)
# The fields were squeezed to 640x640 from a wider frame: isolated erythrocytes measure w/h = 0.60 (median over
# 1,651 in H_100X_C1, per-field p10-p90 0.58-0.64). A crop window this much narrower than tall, resized to a square,
# restores round cells as in C-NMC.
FIELD_X_SCALE = 0.60
# "fixed": one window for the whole subset, CONTEXT x the median unsqueezed box side, so blasts stay larger than
# lymphocytes as in the fixed-size C-NMC crops. "box": CONTEXT x each box's own side (erases that size cue).
# Both values put the pooled median cell share (ellipse in box) at C-NMC's pooled foreground median, 0.143.
WINDOW = "fixed"
CONTEXT = {"fixed": 2.18, "box": 2.2}
NORMAL_EXCLUDED_TYPES = ("CLL",)   # a "lymphocyte" on a CLL slide may be malignant
# A crop whose window shows a cell of the opposite class has an ambiguous label (a Normal crop with a blast in the
# corner): another annotated cell counts as shown when NEIGHBOUR_SHARE of its box lies inside the window.
MALIGNANT_CATEGORIES = ("lymphoblast", "myeloblast", "monoblast", "promonocyte", "abnormal promyelocyte")
OPPOSITE_CATEGORIES = {0: MALIGNANT_CATEGORIES, 1: ("lymphocyte",)}
NEIGHBOUR_SHARE = 0.25
# (field, annotation id) of boxes that frame erythrocytes, not a white cell (COCO and YOLO labels agree; checked by
# eye after a scan of every box for a dark nucleus at the crop centre).
MISPLACED_BOXES = {("35_1_1000_ALL.png", 1356), ("35_1_1000_ALL.png", 1658), ("35_1_1000_ALL.png", 2020)}
# Boxes of one physical cell imaged in two overlapping fields of a slide (SIFT + RANSAC between the fields of each
# slide, overlap NCC > 0.7; data audit): one crop per cell is kept, the least padded.
SAME_CELL_TABLE = "data/processed/splits/leukemiaattri_same_cell.csv"
# One slide holds ~30 % of the lymphoblasts; at most this many ALL crops per slide (least padded first) keep the val
# AUC from being one slide's number.
ALL_PER_SLIDE_CAP = 100


def _find_dir(parent: Path, name: str) -> Path:
    """Child directory of `parent` named `name`, ignoring case (the Drive export is inconsistent)."""
    for child in sorted(parent.iterdir()):
        if child.is_dir() and child.name.lower() == name.lower():
            return child
    raise FileNotFoundError(parent / name)


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def load_boxes(json_paths: Sequence[Path], image_root: Path) -> pd.DataFrame:
    """One row per lymphoblast / lymphocyte box of the given COCO files.

    `train.json` repeats an image entry once per box, so boxes are grouped by file name, and exact repeats of
    (file, category, bbox) are dropped. `image_path` is None when the file is not under `image_root`;
    `conflicting` marks a box whose exact bbox is also annotated with another category.
    """
    on_disk: Dict[str, Path] = {}
    for p in sorted(Path(image_root).rglob("*.png")):
        if p.name in on_disk:
            raise ValueError(f"{p.name} exists twice under {image_root}")
        on_disk[p.name] = p
    rows, seen = [], set()
    for json_path in sorted(Path(p) for p in json_paths):
        coco = json.loads(json_path.read_text(encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        images = {im["id"]: im for im in coco["images"]}
        cats_at: Dict[Tuple[str, Tuple[float, ...]], set] = {}
        for a in coco["annotations"]:
            key = (images[a["image_id"]]["file_name"], tuple(a["bbox"]))
            cats_at.setdefault(key, set()).add(cats[a["category_id"]])
        for a in sorted(coco["annotations"], key=lambda a: a["id"]):
            category = cats[a["category_id"]]
            if category not in CATEGORY_LABELS:
                continue
            im = images[a["image_id"]]
            name = im["file_name"]
            if (name, category, tuple(a["bbox"])) in seen:
                continue
            seen.add((name, category, tuple(a["bbox"])))
            m = IMAGE_RE.match(Path(name).stem)
            if m is None:
                raise ValueError(f"Cannot parse slide / image id from {name}")
            x, y, w, h = a["bbox"]
            rows.append({
                "image_path": str(on_disk[name]) if name in on_disk else None, "file_name": name,
                "slide": m["slide"], "image": m["image"], "leukemia_type": m["type"], "category": category,
                "label": CATEGORY_LABELS[category], "x": x, "y": y, "w": w, "h": h,
                "img_w": im["width"], "img_h": im["height"], "annotation_id": a["id"], "source_json": json_path.name,
                "conflicting": len(cats_at[(name, tuple(a["bbox"]))]) > 1,
            })
    return pd.DataFrame(rows).sort_values(["file_name", "source_json", "annotation_id"], ignore_index=True)


def field_cells(json_paths: Sequence[Path]) -> Dict[str, List[Tuple[int, str, Tuple[float, ...]]]]:
    """(annotation id, category, bbox) of every annotated white cell per field file name ("none" left out)."""
    cells: Dict[str, List[Tuple[int, str, Tuple[float, ...]]]] = {}
    for json_path in sorted(Path(p) for p in json_paths):
        coco = json.loads(json_path.read_text(encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        names = {im["id"]: im["file_name"] for im in coco["images"]}
        for a in coco["annotations"]:
            if cats[a["category_id"]] != "none":
                cells.setdefault(names[a["image_id"]], []).append((a["id"], cats[a["category_id"]], tuple(a["bbox"])))
    return cells


def shows_opposite_class(window: Tuple[int, int, int, int], label: int, annotation_id: int,
                         cells: Sequence[Tuple[int, str, Tuple[float, ...]]]) -> bool:
    """Whether the crop window (x0, y0, width, height) shows NEIGHBOUR_SHARE of a cell of the opposite class."""
    x0, y0, width, height = window
    for aid, category, (bx, by, bw, bh) in cells:
        if aid == annotation_id or category not in OPPOSITE_CATEGORIES[label]:
            continue
        ix = max(0.0, min(x0 + width, bx + bw) - max(x0, bx))
        iy = max(0.0, min(y0 + height, by + bh) - max(y0, by))
        if ix * iy >= NEIGHBOUR_SHARE * bw * bh:
            return True
    return False


def crop_window(
    bbox: Sequence[float], context: float, x_scale: float = FIELD_X_SCALE, ref_side: Optional[float] = None
) -> Tuple[int, int, int, int]:
    """(x0, y0, width, height) of the field window centred on the box: in unsqueezed units a square of side
    `context * ref_side`, or `context * max(w, h)` of the box itself when `ref_side` is None.
    """
    x, y, w, h = bbox
    height = max(1, int(round(context * (ref_side if ref_side is not None else max(w / x_scale, h)))))
    width = max(1, int(round(height * x_scale)))
    return int(round(x + w / 2 - width / 2)), int(round(y + h / 2 - height / 2)), width, height


def crop_cell(
    rgb_uint8: np.ndarray, bbox: Sequence[float], out_size: int = 224, context: float = CONTEXT["box"],
    x_scale: float = FIELD_X_SCALE, ref_side: Optional[float] = None
) -> np.ndarray:
    """Crop centred on the box, edge-replicated where it leaves the field, antialiased resize to a square.

    Args:
        rgb_uint8: [H, W, 3] field.
        bbox: COCO (x, y, w, h).
        out_size: output side.
        context: crop side over the reference side, both in unsqueezed units.
        x_scale: horizontal over vertical pixel scale of the field; 1.0 gives a plain square window.
        ref_side: fixed reference side in unsqueezed px; None uses the box's longer side.
    """
    x0, y0, width, height = crop_window(bbox, context, x_scale, ref_side)
    h, w = rgb_uint8.shape[:2]
    pad = ((max(0, -y0), max(0, y0 + height - h)), (max(0, -x0), max(0, x0 + width - w)), (0, 0))
    padded = np.pad(rgb_uint8, pad, mode="edge")
    crop = padded[y0 + pad[0][0]:y0 + pad[0][0] + height, x0 + pad[1][0]:x0 + pad[1][0] + width]
    if crop.shape[:2] == (out_size, out_size):
        return np.ascontiguousarray(crop)
    t = v2.functional.resize(v2.functional.to_image(crop), [out_size, out_size], antialias=True)
    return t.permute(1, 2, 0).numpy()


def _pad_fraction(
    bbox: Sequence[float], context: float, x_scale: float, ref_side: Optional[float], img_w: int, img_h: int
) -> float:
    """Share of the crop window outside the field, i.e. filled by edge replication."""
    x0, y0, width, height = crop_window(bbox, context, x_scale, ref_side)
    inside = max(0, min(x0 + width, img_w) - max(x0, 0)) * max(0, min(y0 + height, img_h) - max(y0, 0))
    return 1.0 - inside / (width * height)


def _skip_reasons(boxes: pd.DataFrame) -> List[Optional[str]]:
    """Why each box is left out (None = kept): missing image, misplaced box, sliver, cell cut by the field edge,
    lymphocyte on a slide of an excluded type, conflicting label or same-cell repeat.
    """
    reasons: List[Optional[str]] = []
    kept: Dict[Tuple[str, int], List[Tuple[float, ...]]] = {}
    for r in boxes.itertuples():
        bbox = (r.x, r.y, r.w, r.h)
        if pd.isna(r.image_path):
            reasons.append("missing_image")
        elif (r.file_name, r.annotation_id) in MISPLACED_BOXES:
            reasons.append("misplaced_box")
        elif min(r.w, r.h) < MIN_BOX_SIDE:
            reasons.append("degenerate_box")
        elif min(r.x, r.y, r.img_w - r.x - r.w, r.img_h - r.y - r.h) <= EDGE_MARGIN:
            reasons.append("truncated_cell")
        elif r.label == 0 and r.leukemia_type in NORMAL_EXCLUDED_TYPES:
            reasons.append("excluded_slide_type")
        elif r.conflicting:
            reasons.append("conflicting_label")
        elif any(_iou(bbox, other) > DUPLICATE_IOU for other in kept.get((r.file_name, r.label), [])):
            reasons.append("duplicate_box")
        else:
            kept.setdefault((r.file_name, r.label), []).append(bbox)
            reasons.append(None)
    return reasons


def build_val_cohort(
    subset_dir: Path, out_dir: Path, out_size: int = 224, context: Optional[float] = None,
    x_scale: float = FIELD_X_SCALE, window: str = WINDOW
) -> pd.DataFrame:
    """Write `out_dir/{Normal,ALL}/<slide>_<image>_<annid>.png` and `out_dir/manifest.csv`; returns the manifest.

    Deterministic and idempotent: identical existing crops are not rewritten and stale PNGs are removed, so
    `out_dir` always holds exactly the manifest.

    Args:
        subset_dir: e.g. data/raw/LeukemiaAttri/H_100X_C1.
        out_dir: output folder.
        out_size: crop side on disk.
        context: window side over the reference side; None takes `CONTEXT[window]`.
        x_scale: horizontal over vertical pixel scale of the fields.
        window: "fixed" (reference = median unsqueezed side of all the subset's boxes) or "box" (each box's own).
    """
    if window not in CONTEXT:
        raise ValueError(f"window must be one of {list(CONTEXT)}, got {window!r}")
    context = CONTEXT[window] if context is None else context
    subset_dir, out_dir = Path(subset_dir), Path(out_dir)
    json_paths = sorted(_find_dir(subset_dir, "json_labels").glob("*.json"))
    boxes = load_boxes(json_paths, _find_dir(subset_dir, "Images"))
    ref_side = float(np.median(np.maximum(boxes["w"] / x_scale, boxes["h"]))) if window == "fixed" else None
    boxes["skip"] = _skip_reasons(boxes)
    cells = field_cells(json_paths)
    for i, r in boxes[boxes["skip"].isna()].iterrows():
        window = crop_window((r.x, r.y, r.w, r.h), context, x_scale, ref_side)
        if shows_opposite_class(window, r.label, r.annotation_id, cells.get(r.file_name, [])):
            boxes.at[i, "skip"] = "opposite_class_neighbour"
    boxes["pad_fraction"] = [
        round(_pad_fraction((r.x, r.y, r.w, r.h), context, x_scale, ref_side, r.img_w, r.img_h), 4)
        for r in boxes.itertuples()
    ]
    same = pd.read_csv(abs_path(SAME_CELL_TABLE))
    group_of = {(r.file, r.x, r.y, r.w, r.h): r.cell_group for r in same.itertuples()}
    boxes["cell_group"] = [group_of.get((r.file_name, r.x, r.y, r.w, r.h)) for r in boxes.itertuples()]
    kept = boxes[boxes["skip"].isna()].sort_values(["pad_fraction", "annotation_id"])
    repeats = kept[kept["cell_group"].notna() & kept.duplicated("cell_group")].index
    boxes.loc[repeats, "skip"] = "same_cell_other_field"
    kept = boxes[boxes["skip"].isna() & (boxes["label"] == 1)].sort_values(["pad_fraction", "annotation_id"])
    boxes.loc[kept[kept.groupby("slide").cumcount() >= ALL_PER_SLIDE_CAP].index, "skip"] = "slide_cap"
    skipped = boxes["skip"].value_counts().to_dict()
    boxes = boxes[boxes["skip"].isna()].copy()
    boxes["file"] = [f"{CLASS_NAMES[r.label]}/{r.slide}_{r.image}_{r.annotation_id}.png" for r in boxes.itertuples()]
    if boxes["file"].duplicated().any():
        raise ValueError(f"Output name collision: {boxes.loc[boxes['file'].duplicated(), 'file'].tolist()[:5]}")

    for cls in CLASS_NAMES:
        (out_dir / cls).mkdir(parents=True, exist_ok=True)
    wanted = set(boxes["file"])
    for stale in sorted(p for cls in CLASS_NAMES for p in (out_dir / cls).glob("*.png")):
        if f"{stale.parent.name}/{stale.name}" not in wanted:
            stale.unlink()
    written = 0
    for path, group in boxes.groupby("image_path", sort=True):
        field = np.asarray(Image.open(path).convert("RGB"))
        for r in group.itertuples():
            crop = crop_cell(field, (r.x, r.y, r.w, r.h), out_size, context, x_scale, ref_side)
            dst = out_dir / r.file
            if dst.exists() and np.array_equal(np.asarray(Image.open(dst).convert("RGB")), crop):
                continue
            Image.fromarray(crop).save(dst, format="PNG", compress_level=9)
            written += 1

    boxes["class_name"] = [CLASS_NAMES[v] for v in boxes["label"]]
    boxes["source_image"] = [Path(p).relative_to(subset_dir).as_posix() for p in boxes["image_path"]]
    boxes["window_h"] = [crop_window((r.x, r.y, r.w, r.h), context, x_scale, ref_side)[3] for r in boxes.itertuples()]
    manifest = boxes[[
        "file", "label", "class_name", "category", "slide", "leukemia_type", "source_image", "x", "y", "w", "h",
        "annotation_id", "source_json", "window_h", "pad_fraction",
    ]].sort_values("file", ignore_index=True)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    reference = f"median box side {ref_side:.1f} px" if ref_side is not None else "each box side"
    print(f"[OK] {len(manifest)} crops -> {out_dir} ({written} written, {len(manifest) - written} unchanged); "
          f"window {context} x {reference}, x_scale {x_scale}; skipped {skipped or 'none'}")
    return manifest
