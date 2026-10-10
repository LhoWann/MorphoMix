"""Figures: the augmentation panels (`python main.py figures`) and the paper figures drawn from results_dir/tables
and results_dir/predictions (`python main.py paperfigs`): the study overview (protocol, MorphoMix pipeline,
whole-field inference) and the result figures."""
import argparse
import glob
import json
import os
import textwrap
from functools import partial
from typing import Callable, Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
from PIL import Image

from src.augmentations.morpho_mix import apply_morpho_mix
from src.augmentations.stain import from_unit, match_regions, to_unit
from src.augmentations.stain_baselines import apply_hed_jitter, apply_randstainna, apply_stain_mixup, foreground
from src.augmentations.style_bank import StyleBank
from src.augmentations.transforms import IMAGENET_MEAN, IMAGENET_STD, denormalize_image, get_train_transforms
from src.cam.blast_prior import extract_blast_cell_prior
from src.cam.metrics import central_cell_mask
from src.cam.prior_only import PriorOnlyCAM
from src.datasets.cells import cell_mask
from src.evaluation.field_inference import detect_cells
from src.utils import figstyle
from src.utils.config import abs_path, load_config


C_INPUT = figstyle.COLOURS["grey"]
C_BASE = figstyle.COLOURS["blue"]
C_OURS = figstyle.COLOURS["orange"]

TEXT_MUTED = "#4A4A4A"
TEXT_DARK = "#1A1A1A"


def to_tensor(img_np: np.ndarray) -> torch.Tensor:
    """uint8 RGB [224, 224, 3] -> normalised tensor [1, 3, 224, 224]."""
    img_f = img_np.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    norm = (img_f - mean) / std
    return torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float()


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Normalised tensor [1, 3, H, W] -> uint8 RGB [H, W, 3]."""
    return denormalize_image(tensor[0])


def rect(fig, x, y, w, h):
    """Axes rectangle from inches (origin bottom-left) to the figure fractions Matplotlib wants."""
    fw, fh = fig.get_size_inches()
    return [x / fw, y / fh, w / fw, h / fh]


def fig_xy(fig, x, y):
    """A point in inches, as figure fractions."""
    fw, fh = fig.get_size_inches()
    return x / fw, y / fh


def draw_image(fig, x, y, side, img, letter: str, edge: str):
    """One square image cell: the picture, a group-coloured frame, and a lettered corner badge."""
    ax = fig.add_axes(rect(fig, x, y, side, side))
    ax.imshow(img, interpolation="lanczos")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(edge)
        spine.set_linewidth(1.3)
    # Dark text on a light badge: white text in a PDF reads as hidden text to plagiarism checkers
    ax.text(0.04, 0.96, letter, transform=ax.transAxes, ha="left", va="top",
            fontsize=7.5, fontweight="bold", color=TEXT_DARK,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=edge, linewidth=0.8, alpha=0.92))
    return ax


DIFF_FULL_SCALE = 25.0  # CIELAB delta E at the top of the colour map: |difference| amplified x4 (delta E spans ~100)
DIFF_CMAP = "magma"


def colour_difference(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-pixel CIELAB delta E (CIE76) between two uint8 RGB images, as an RGB magma map (black = unchanged)."""
    lab = [cv2.cvtColor(x.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB) for x in (img, ref)]
    delta = np.linalg.norm(lab[0] - lab[1], axis=-1)
    return np.uint8(255 * plt.get_cmap(DIFF_CMAP)(np.clip(delta / DIFF_FULL_SCALE, 0, 1))[..., :3])


def draw_band(fig, x0, x1, y, text: str, colour: str, note: str = ""):
    """A group header."""
    fw, fh = fig.get_size_inches()
    fig.add_artist(Rectangle(fig_xy(fig, x0, y), (x1 - x0) / fw, 0.004,
                             transform=fig.transFigure, facecolor=colour,
                             edgecolor="none", zorder=3))
    # A group narrower than its own label
    anchor_x, align = (x0, "left") if 0.062 * len(text) <= (x1 - x0) else (x1, "right")
    fig.text(*fig_xy(fig, anchor_x, y + 0.055), text, ha=align, va="bottom",
             fontsize=8.5, fontweight="bold", color=colour)
    if note:
        fig.text(*fig_xy(fig, x1, y + 0.055), note, ha="right", va="bottom",
                 fontsize=7, color=TEXT_MUTED)


LINE_SPACING = 1.5


def wrap_lines(lines, width, fontsize):
    """Break caption paragraphs to the width actually available, in rendered lines."""
    chars = max(20, int(width * 72 / (0.56 * fontsize)))
    out = []
    for line in lines:
        out.extend(textwrap.wrap(line, chars))
    return out


def caption_height(lines, fontsize):
    """Height in inches of a block returned by `wrap_lines`."""
    return len(lines) * fontsize * LINE_SPACING / 72.0


def draw_caption(fig, x, y_top, lines, fontsize=7.2):
    """Renders the already-wrapped `lines`, top-aligned at `y_top`."""
    fig.text(*fig_xy(fig, x, y_top), "\n".join(lines), ha="left", va="top",
             fontsize=fontsize, color=TEXT_MUTED, linespacing=LINE_SPACING)


def draw_arrow(fig, x_from, x_to, y, colour="#9A9A9A"):
    """A flow arrow between two panels."""
    fig.add_artist(FancyArrowPatch(fig_xy(fig, x_from, y), fig_xy(fig, x_to, y),
                                   transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=9, lw=1.0, color=colour,
                                   shrinkA=0, shrinkB=0, zorder=4))


def draw_strip(stages, title: str, caption: List[str], path: str, arrows: bool = True) -> None:
    """One row of panels in coloured groups; `stages` = [(colour, group title, [(img, letter, name), ...]), ...]."""
    fig_w = figstyle.WIDTH_DOUBLE
    margin, gap = 0.22, 0.13
    n_panels = sum(len(cells) for _, _, cells in stages)
    side = (fig_w - 2 * margin - gap * (n_panels - 1)) / n_panels
    caption = wrap_lines(caption, fig_w - 2 * margin, 7.0)
    # Vertical budget, bottom-up
    name_h, band_h, title_h = 0.22, 0.26, 0.50
    img_bottom = 0.10 + caption_height(caption, 7.0) + name_h
    fig_h = img_bottom + side + band_h + title_h
    fig = plt.figure(figsize=(fig_w, fig_h))

    idx = 0
    for colour, group, cells in stages:
        x_start = margin + idx * (side + gap)
        x_end = x_start + len(cells) * side + (len(cells) - 1) * gap
        draw_band(fig, x_start, x_end, img_bottom + side + 0.10, group, colour)
        for img, letter, name in cells:
            x = margin + idx * (side + gap)
            draw_image(fig, x, img_bottom, side, img, letter, colour)
            fig.text(*fig_xy(fig, x + side / 2, img_bottom - 0.08), name,
                     ha="center", va="top", fontsize=7.0, color=TEXT_DARK)
            if arrows and idx < n_panels - 1:
                draw_arrow(fig, x + side + 0.02, x + side + gap - 0.02, img_bottom + side / 2)
            idx += 1

    fig.suptitle(title, fontsize=10.5, fontweight="bold", y=1 - 0.24 / fig_h)
    draw_caption(fig, margin, img_bottom - name_h - 0.04, caption, fontsize=7.0)
    figstyle.finish(fig, path)


def farthest_rows(points: torch.Tensor, k: int) -> List[int]:
    """k rows of `points` [N, D] by farthest-point sampling from row 0, so the references span the bank."""
    chosen = [0]
    dist = (points - points[0]).norm(dim=1)
    for _ in range(k - 1):
        chosen.append(int(dist.argmax()))
        dist = torch.minimum(dist, (points - points[chosen[-1]]).norm(dim=1))
    return chosen


def generate_augmentation_figures(
    normal_path: str = "data/processed/train/Normal/CNMC_fold_0_UID_H11_1_1_hem.png",
    all_path: str = "data/processed/train/ALL/CNMC_fold_0_UID_11_10_1_all.png",
    output_dir: Optional[str] = None
):
    """Generates the publication panels for every augmentation arm (default: results_dir/figures)."""
    cfg = load_config()
    output_dir = output_dir or abs_path(os.path.join(cfg["results_dir"], "figures", "augmentations"))
    os.makedirs(output_dir, exist_ok=True)
    figstyle.apply()
    # These figures are positioned in inches
    plt.rcParams["savefig.bbox"] = None

    np_a = np.array(Image.open(normal_path).convert("RGB").resize((224, 224)))
    np_b = np.array(Image.open(all_path).convert("RGB").resize((224, 224)))
    batch_images = torch.cat([to_tensor(np_a), to_tensor(np_b)], dim=0)  # [2, 3, 224, 224]

    # Azure-B prior, min-max normalised per image as in training (PriorOnlyCAM): what every component thresholds
    batch_cams = PriorOnlyCAM(prior_threads=1).generate(batch_images)
    soft_prior_b = batch_cams[1, 0].numpy()
    threshold = float(cfg["morpho_threshold"])
    bank = StyleBank.load(cfg["style_bank"]["path"])  # both built by `prepare`
    with open(abs_path(cfg["randstainna_stats"]), encoding="utf-8") as f:
        randstainna_stats = json.load(f)
    with np.load(abs_path(cfg["style_bank"]["path"])) as z:
        target_stains = torch.from_numpy(z["stain_matrix"])
    class_names = cfg["phase1"]["class_names"]

    # every arm is drawn on the ALL cell (b); no arm mixes labels
    torch.manual_seed(42)
    np_basic = to_numpy(get_train_transforms(img_size=224)(Image.fromarray(np_b)).unsqueeze(0))
    fg = foreground(batch_images)

    def baseline(fn, *args, **kwargs) -> np.ndarray:
        torch.manual_seed(42)
        return to_numpy(fn(batch_images, fg, *args, **kwargs)[1:2])

    baselines = {
        "hed_jitter": baseline(apply_hed_jitter, sigma=float(cfg["hed_sigma"])),
        "randstainna": baseline(apply_randstainna, randstainna_stats, std_hyper=float(cfg["randstainna_std_hyper"])),
        "stain_mixup": baseline(apply_stain_mixup, target_stains),
    }

    # MorphoMix on the ALL cell (sample 1), components accumulated in execution order; labels stay hard. C1 in its
    # final MLL23 form; the RandStainNA form is ablation arm e
    def morpho(**switches):
        torch.manual_seed(42)
        kwargs = {"appearance_prob": 1.0, "rsn_prob": 0.0, "small_cell_lowdetail_prob": 0.0, **switches}
        out, masks = apply_morpho_mix(
            batch_images, batch_cams, threshold=threshold, bank=bank, appearance_alpha=(1.0, 1.0),
            virtual_template_prob=0.0, acquisition_prob=0.0, small_cell_px=tuple(cfg["small_cell_px"]),
            background_prob=1.0, background_rbc=tuple(cfg["background_rbc"]),
            rsn_stats=randstainna_stats, rsn_std_hyper=float(cfg["randstainna_std_hyper"]), **kwargs)
        return to_numpy(out[1:2]), masks

    np_c1, _ = morpho(use_background=False)
    np_c12, _ = morpho(small_cell_prob=1.0, use_background=False)
    np_c12_low, _ = morpho(small_cell_prob=1.0, small_cell_lowdetail_prob=1.0, use_background=False)
    np_c123, mask_c123 = morpho(small_cell_prob=1.0)
    np_rsn, _ = morpho(rsn_prob=1.0, appearance_prob=0.0, use_background=False)

    # auxiliary training cells (aux_train), first file of each class; absent when they were never built
    aux = abs_path(cfg["aux_train"]["dir"])
    aux_cells = [(np.array(Image.open(sorted(glob.glob(os.path.join(aux, c, "*.png")))[0]).convert("RGB")), c)
                 for c in class_names if glob.glob(os.path.join(aux, c, "*.png"))]

    singles = {"01_input_normal": np_a, "02_input_all": np_b, "03_basic": np_basic,
               "04_hed_jitter": baselines["hed_jitter"], "05_randstainna": baselines["randstainna"],
               "06_stain_mixup": baselines["stain_mixup"], "07_morphomix_c1_stain": np_c1,
               "08_morphomix_c1_c2p_small_cell": np_c12, "09_morphomix_c1_c2p_lowdetail": np_c12_low,
               "10_morphomix_c1_c2p_c3": np_c123}
    for name, img in singles.items():
        Image.fromarray(img).save(os.path.join(output_dir, f"{name}.png"))

    # comparison figure: training cells, the baselines and MorphoMix on cell (b); the caption belongs to the paper
    label = figstyle.ARM_LABEL
    letters = iter(f"({chr(c)})" for c in range(ord("a"), ord("z")))
    # rows: (colour, title, note, cells, with a colour-difference map to cell (b) under each panel)
    rows = [
        (C_INPUT, "Training cells", "C-NMC and auxiliary Bodzas cells", [
            (np_a, next(letters), f"C-NMC {class_names[0]}"),
            (np_b, next(letters), f"C-NMC {class_names[1]}"),
            *[(img, next(letters), f"Bodzas {c} (auxiliary)") for img, c in aux_cells],
        ], False),
        (C_BASE, "Baselines", "applied to cell (b)", [
            (np_basic, next(letters), "Basic"),
            (baselines["hed_jitter"], next(letters), label["hed_jitter"]),
            (baselines["randstainna"], next(letters), label["randstainna"]),
            (baselines["stain_mixup"], next(letters), label["stain_mixup"]),
        ], True),
        (C_OURS, "MorphoMix", "applied to cell (b), one Azure-B cell mask", [
            (np_c1, next(letters), "$C_1$ in-cell stain"),
            (np_c12, next(letters), "$C_1 + C_2$ small, before $C_3$"),
            (np_c12_low, next(letters), "$C_1 + C_2$ low detail"),
            (np_c123, next(letters), "$C_1 + C_2 + C_3$"),
        ], True),
    ]

    fig_w = figstyle.WIDTH_DOUBLE
    margin_l, margin_r = 0.20, 0.20
    n_cols = max(len(cells) for _, _, _, cells, _ in rows)
    col_w = (fig_w - margin_l - margin_r) / n_cols
    # Vertical budget of one group, in inches
    band_h, band_gap, img_side, name_drop, row_gap = 0.14, 0.12, 1.40, 0.24, 0.14
    diff_gap, diff_side = 0.06, 0.78
    top, bottom = 0.20, 0.04
    blocks = [band_h + band_gap + img_side + name_drop + row_gap + with_diff * (diff_gap + diff_side)
              for *_, with_diff in rows]
    fig_h = top + sum(blocks) + bottom
    fig = plt.figure(figsize=(fig_w, fig_h))

    rule_y = fig_h - top - band_h
    for (colour, title, note, cells, with_diff), block in zip(rows, blocks):
        draw_band(fig, margin_l, fig_w - margin_r, rule_y, title, colour, note)
        img_bottom = rule_y - band_gap - img_side
        name_y = img_bottom - with_diff * (diff_gap + diff_side)
        for c, (img, letter, name) in enumerate(cells):
            x_centre = margin_l + (c + 0.5 + (n_cols - len(cells)) / 2) * col_w
            ax = draw_image(fig, x_centre - img_side / 2, img_bottom, img_side, img, letter, colour)
            if img is np_c12:  # an intermediate: a small cell always gets the C3 background before the network
                for spine in ax.spines.values():
                    spine.set_linestyle((0, (3, 2)))
            if with_diff:
                diff = fig.add_axes(rect(fig, x_centre - diff_side / 2, name_y, diff_side, diff_side))
                diff.imshow(colour_difference(img, np_b), interpolation="lanczos")
                diff.set_xticks([])
                diff.set_yticks([])
                for spine in diff.spines.values():
                    spine.set_color(colour)
                    spine.set_linewidth(0.8)
            fig.text(*fig_xy(fig, x_centre, name_y - 0.05), name,
                     ha="center", va="top", fontsize=8, color=TEXT_DARK)
        if with_diff:
            bar = fig.add_axes(rect(fig, x_centre + diff_side / 2 + 0.10, name_y, 0.06, diff_side))
            scale = plt.cm.ScalarMappable(plt.Normalize(0, DIFF_FULL_SCALE), DIFF_CMAP)
            fig.colorbar(scale, cax=bar, ticks=[0, DIFF_FULL_SCALE])
            bar.tick_params(labelsize=6.5, length=2, pad=1.5)
            bar.set_yticklabels(["0", f"{DIFF_FULL_SCALE:.0f}"])
            bar.set_ylabel("$\\Delta E$", fontsize=7, labelpad=1)
        rule_y -= block

    panel_path = os.path.join(output_dir, "all_augmentations_comparison")
    figstyle.finish(fig, panel_path)

    # pipeline figure
    heatmap_rgb = cv2.applyColorMap(np.uint8(255 * soft_prior_b), cv2.COLORMAP_MAGMA)
    heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)
    overlay_heat = np.clip(0.55 * heatmap_rgb + 0.45 * np_b, 0, 255).astype(np.uint8)

    mask_hard = cell_mask(np_b, threshold).astype(np.uint8) * 255  # the mask apply_morpho_mix builds
    mask_hard_3c = np.stack([mask_hard] * 3, axis=-1)

    # Soft-edge alpha
    kernel_size = 5
    padding = kernel_size // 2
    t_mask = torch.from_numpy(mask_hard.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    soft_t_mask = torch.nn.functional.avg_pool2d(t_mask, kernel_size=kernel_size, stride=1, padding=padding)
    soft_t_mask = torch.minimum(soft_t_mask, t_mask)
    np_soft_mask = (soft_t_mask[0, 0].numpy() * 255.0).astype(np.uint8)
    np_soft_mask_3c = np.stack([np_soft_mask] * 3, axis=-1)

    # Synthetic smear background (component C3)
    from src.augmentations.background import synthesize_backgrounds
    torch.manual_seed(7)
    synth_bg = synthesize_backgrounds(1, np_b.shape[0], np_b.shape[1], device=torch.device("cpu"))
    np_synth_bg = (synth_bg[0].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    np_mask_final = (np.asarray(mask_c123[1, 0]) > 0.5).astype(np.uint8) * 255

    pipeline_path = os.path.join(output_dir, "morpho_mix_configs_ablation")
    draw_strip([
        (C_INPUT, "Stage 1 - one cell mask", [
            (overlay_heat, "(a)", "Cell prior"),
            (mask_hard_3c, "(b)", f"Mask, $\\tau = {threshold:.2f}$"),
            (np_soft_mask_3c, "(c)", "Feather"),
        ]),
        (C_BASE, "Stage 2 - components", [
            (np_c1, "(d)", "$C_1$ stain"),
            (np_c12, "(e)", "$C_2$ small cell"),
            (np_synth_bg, "(f)", "$C_3$ context"),
            (np_rsn, "(g)", "RandStainNA $C_1$ (arm e)"),
        ]),
        (C_OURS, "Stage 3 - network input", [
            (np_c123, "(h)", "$C_1 + C_2 + C_3$"),
            (np.stack([np_mask_final] * 3, axis=-1), "(i)", "Mask of (h)"),
        ]),
    ], "MorphoMix pipeline: one mask, reused by every component", [
        # Plain prose in the caption, no inline maths
        "(a)-(b) the Azure-B prior and its binary mask, the one mask every component uses. (c) a 5x5 box "
        "filter clamped back to the hard mask, so the C3 feather never spreads into the surround. "
        "(d) nucleus and cytoplasm matched in Lab to a random MLL23 cell (or a virtual template). "
        "(e) the cell shrunk to field scale (24-48 px) at a random position. (f) procedural plasma and "
        "erythrocytes; no real image is ever pasted in. (g) the RandStainNA form of C1 (ablation arm e): the cell "
        "recoloured with a template fitted to the training cells. (i) the mask after C2, which C3 never changes.",
    ], pipeline_path)

    print(f"[OK] Augmentation figures written to: {output_dir}")
    print(f"  * Master comparison: {panel_path}.png / .pdf")
    print(f"  * Pipeline strip:    {pipeline_path}.png / .pdf")

    # C1 diversity: one cell against bank references that span the bank, and one virtual template
    n_refs = 6
    picked = farthest_rows(bank.mean[:, 0], n_refs)  # on the nucleus Lab mean
    torch.manual_seed(42)
    v_mean, v_sd = bank.virtual(1, torch.device("cpu"))
    ref_mean, ref_sd = torch.cat([bank.mean[picked], v_mean]), torch.cat([bank.sd[picked], v_sd])
    rgb = to_unit(batch_images[1:2]).expand(n_refs + 1, -1, -1, -1)
    cell = torch.from_numpy(mask_hard / 255.0).float().view(1, 1, 224, 224).expand(n_refs + 1, -1, -1, -1)
    matched = from_unit(match_regions(rgb, cell, ref_mean, ref_sd))
    renders = [to_numpy(matched[i:i + 1]) for i in range(n_refs + 1)]
    letters = [f"({chr(ord('b') + i)})" for i in range(n_refs + 1)]
    diversity_path = os.path.join(output_dir, "morpho_mix_c1_diversity")
    draw_strip([
        (C_INPUT, "Input", [(np_b, "(a)", "C-NMC cell")]),
        (C_BASE, f"MLL23 bank references ({len(bank):,} cells)",
         [(img, letter, f"bank row {row}") for img, letter, row in zip(renders, letters, picked)]),
        (C_OURS, "Virtual", [(renders[-1], letters[-1], "template")]),
    ], "MorphoMix C1: one cell in many MLL23 stains", [
        f"Cell (a) with its nucleus and cytoplasm matched in Lab (Reinhard, per region, full strength) to {n_refs} "
        "MLL23 bank cells chosen by farthest-point sampling on the nucleus Lab mean, and to one virtual template "
        "drawn from the Gaussian fitted to the bank. Pixels outside the cell mask never change; MLL23 labels are "
        "never read.",
    ], diversity_path, arrows=False)
    print(f"  * C1 diversity:      {diversity_path}.png / .pdf")


# --------------------------------------------------------------------------------------------------------------------
# Paper figures
# --------------------------------------------------------------------------------------------------------------------

FOUNDATION_ARMS = ("dinobloom_s", "dinobloom_b")  # scripts/foundation.py MODELS
BG_REF = "#B03A2E"

INK = "#1A1A1A"
MUTED = "#555555"
ROLE = {"train": "#3B3B3B", "select": figstyle.COLOURS["blue"], "test": figstyle.COLOURS["darkorange"],
        "ref": figstyle.COLOURS["lightgrey"]}
FILL = {"train": "#EFEFEF", "select": "#E3EEF7", "test": "#FBE9DD", "ref": "#F5F5F5", "ours": "#FFF1E3"}

EXAMPLES = {
    "cnmc": "data/processed/train/ALL/CNMC_fold_0_UID_11_10_1_all.png",
    "bodzas": "data/processed/train_aux_bodzas2023/ALL/bodzas2023_Lymphoblast_0.png",
    "mll23": "data/raw/MLL23/lymphocyte/lymphocyte/lymphocyte_0001.TIF",
    "leukemiaattri": "data/processed/val/ALL/5_57_7129.png",
    "allidb2": "data/processed/test_allidb2/ALL/ALLIDB_Im001_1.png",
}


def rgb(path: str, side: int = 224) -> np.ndarray:
    return np.asarray(Image.open(abs_path(path)).convert("RGB").resize((side, side), Image.BICUBIC))


class Canvas:
    """One full-figure axes in inch coordinates (origin bottom-left), for diagram-style figures."""

    def __init__(self, width: float, height: float):
        self.fig = plt.figure(figsize=(width, height))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.axis("off")

    def image(self, img: np.ndarray, x: float, y: float, side: float, edge: str = "#BBBBBB", lw: float = 0.6):
        self.ax.imshow(img, extent=(x, x + side, y, y + side), interpolation="lanczos", zorder=2)
        self.ax.add_patch(plt.Rectangle((x, y), side, side, fill=False, ec=edge, lw=lw, zorder=3))

    def box(self, x: float, y: float, w: float, h: float, fc: str, ec: str = "none", lw: float = 0.8,
            radius: float = 0.06):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                         fc=fc, ec=ec, lw=lw, zorder=1))

    def text(self, x: float, y: float, s: str, size: float = 7.0, colour: str = INK, weight: str = "normal",
             ha: str = "center", va: str = "center", **kw):
        self.ax.text(x, y, s, fontsize=size, color=colour, fontweight=weight, ha=ha, va=va, zorder=4, **kw)

    def arrow(self, x0: float, y0: float, x1: float, y1: float, colour: str = "#7A7A7A", lw: float = 1.0,
              head: float = 8):
        self.ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=head, lw=lw,
                                          color=colour, shrinkA=0, shrinkB=0, zorder=4))

    def panel_title(self, x: float, y: float, letter: str, title: str):
        self.text(x, y, f"({letter})", size=8.5, weight="bold", ha="left", va="bottom")
        self.text(x + 0.27, y, title, size=8.5, weight="bold", ha="left", va="bottom")


def prior_map(img: np.ndarray) -> np.ndarray:
    soft = extract_blast_cell_prior(img)[0]
    soft = (soft - soft.min()) / max(float(soft.max() - soft.min()), 1e-8)
    return cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * soft), cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB)


def mask_view(img: np.ndarray, threshold: float) -> np.ndarray:
    m = cell_mask(img, threshold) > 0
    out = np.full_like(img, 28)
    out[m] = (255, 255, 255)
    return out


def fig_overview(cfg: Dict, out: str) -> Optional[str]:
    """Study design (one laboratory per role), the MorphoMix training pipeline and the cell-by-cell field
    inference, with real images from every cohort."""
    aug_dir = abs_path(os.path.join(cfg["results_dir"], "figures", "augmentations"))
    stages = ["02_input_all", "07_morphomix_c1_stain", "08_morphomix_c1_c2p_small_cell", "10_morphomix_c1_c2p_c3"]
    if not all(os.path.exists(os.path.join(aug_dir, f"{s}.png")) for s in stages):
        print("  run `python main.py figures` first (augmentation panels)")
        return None
    aug = {s: rgb(os.path.join(aug_dir, f"{s}.png")) for s in stages}
    counts = {k: len(glob.glob(abs_path(f"data/processed/{d}/*/*.png")))
              for k, d in [("cnmc", "train"), ("bodzas", "train_aux_bodzas2023"), ("leukemiaattri", "val"),
                           ("allidb2", "test_allidb2"), ("aria", "test_aria")]}

    W, H = figstyle.WIDTH_DOUBLE, 6.55
    c = Canvas(W, H)

    # (a) study design: one laboratory per role
    top = H - 0.12
    c.panel_title(0.05, top - 0.2, "a", "Study design")
    cards = [
        ("train", "C-NMC 2019", "AIIMS New Delhi", f"{counts['cnmc']:,} single cells", "cnmc"),
        ("train", "Bodzas 2023", "Ostrava (auxiliary)", f"{counts['bodzas']:,} single cells", "bodzas"),
        ("ref", "MLL23", "Munich", "stain statistics only", "mll23"),
        ("select", "LeukemiaAttri", "Lahore", f"{counts['leukemiaattri']:,} cell crops", "leukemiaattri"),
        ("test", "ALL-IDB2", "Milan", f"{counts['allidb2']:,} single cells", "allidb2"),
        ("test", "Aria", "Tehran", f"{counts['aria']:,} whole fields", "aria"),
    ]
    groups = [("train", "Training", 0, 2), ("ref", "Stain references", 2, 3), ("select", "Model selection", 3, 4),
              ("test", "External testing", 4, 6)]
    margin, gap = 0.08, 0.07
    card_w = (W - 2 * margin - gap * (len(cards) - 1)) / len(cards)
    card_h, img_side = 1.62, 0.84
    y_card = top - 0.48 - card_h
    for role, label, i0, i1 in groups:
        x0 = margin + i0 * (card_w + gap)
        x1 = margin + i1 * (card_w + gap) - gap
        c.ax.plot([x0, x1], [y_card + card_h + 0.08] * 2, color=ROLE[role], lw=2.2, solid_capstyle="butt")
        c.text((x0 + x1) / 2, y_card + card_h + 0.17, label, size=7.5, weight="bold", colour=ROLE[role])
    for i, (role, name, place, n, key) in enumerate(cards):
        x = margin + i * (card_w + gap)
        c.box(x, y_card, card_w, card_h, FILL[role])
        img = rgb(f"data/processed/test_aria/ALL/{example_field(cfg)[0]}" if key == "aria" else EXAMPLES[key])
        c.image(img, x + (card_w - img_side) / 2, y_card + card_h - img_side - 0.08, img_side)
        c.text(x + card_w / 2, y_card + 0.50, name, size=7.2, weight="bold")
        c.text(x + card_w / 2, y_card + 0.32, place, size=6.4, colour=MUTED)
        c.text(x + card_w / 2, y_card + 0.15, n, size=6.4, colour=MUTED)

    # (b) MorphoMix: one Azure-B cell mask drives every component, on the GPU batch; the label is never changed
    top_b = y_card - 0.22
    c.panel_title(0.05, top_b - 0.2, "b", "MorphoMix")
    tau = float(cfg["morpho_threshold"])
    cell = aug["02_input_all"]
    steps = [
        (cell, "input cell $x$", "label $y$", ROLE["train"]),
        (prior_map(cell), "Azure-B prior", "soft cell envelope", "#7A7A7A"),
        (mask_view(cell, tau), "cell mask $M$", f"prior $\\geq \\tau = {tau:.2f}$", "#7A7A7A"),
        (aug["07_morphomix_c1_stain"], "$C_1$ in-cell stain",
         f"nucleus, cytoplasm\nto an MLL23 cell\np = {cfg['appearance_prob']}", figstyle.COLOURS["orange"]),
        (aug["08_morphomix_c1_c2p_small_cell"], "$C_2$ small cell",
         f"to {cfg['small_cell_px'][0]}-{cfg['small_cell_px'][1]} px,\nor low detail\np = {cfg['small_cell_prob']}",
         figstyle.COLOURS["orange"]),
        (aug["10_morphomix_c1_c2p_c3"], "$C_3$ background",
         f"synthetic smear\noutside $M$\np = {cfg['background_prob']:.3f}", figstyle.COLOURS["orange"]),
    ]
    side, step_gap = 0.76, 0.20
    model_w = W - 2 * margin - len(steps) * side - len(steps) * step_gap
    y_img = top_b - 0.62 - side
    x_ours = margin + 3 * (side + step_gap) - 0.07
    c.box(x_ours, y_img - 0.64, 3 * side + 2 * step_gap + 0.14, side + 0.92, FILL["ours"],
          ec=figstyle.COLOURS["paleorange"], lw=0.8)
    c.text(x_ours + 0.07, y_img + side + 0.15, "MorphoMix components (label unchanged)", size=6.3,
           weight="bold", colour=figstyle.COLOURS["darkorange"], ha="left")
    for i, (img, name, note, edge) in enumerate(steps):
        x = margin + i * (side + step_gap)
        c.image(img, x, y_img, side, edge=edge, lw=1.1)
        c.text(x + side / 2, y_img - 0.11, name, size=6.6, weight="bold")
        if note:
            c.text(x + side / 2, y_img - 0.24, note, size=5.7, colour=MUTED, va="top", linespacing=1.25)
        c.arrow(x + side + 0.03, y_img + side / 2, x + side + step_gap - 0.03, y_img + side / 2)
    x_model = margin + len(steps) * (side + step_gap)
    c.box(x_model, y_img + 0.04, model_w, side - 0.08, "#E6E6E6", ec="#9A9A9A", lw=0.7)
    c.text(x_model + model_w / 2, y_img + side / 2 + 0.15, "ConvNeXt V2", size=6.6, weight="bold")
    c.text(x_model + model_w / 2, y_img + side / 2 - 0.01, "Atto", size=6.6, weight="bold")
    c.text(x_model + model_w / 2, y_img + side / 2 - 0.19, "weighted CE on $y$", size=5.7, colour=MUTED)

    # (c) whole fields are scored cell by cell with the single-cell model
    top_c = y_img - 0.72
    c.panel_title(0.05, top_c - 0.2, "c", "Whole-field inference (Aria)")
    inf = cfg["inference"]["cells"]
    name, probs = example_field(cfg)
    field = rgb(f"data/processed/test_aria/ALL/{name}")
    cells = detect_cells(field, float(inf["min_radius"]), min_inside=float(inf["min_inside"]),
                         min_nucleus=float(inf["min_nucleus"]))
    if len(cells) != len(probs):
        raise RuntimeError(f"{name}: {len(cells)} detections here, {len(probs)} scored cells in the predictions")
    drawn = field.copy()
    for k, (x, y, r) in enumerate(cells):
        cv2.circle(drawn, (int(round(x)), int(round(y))), int(round(r)) + 3, (255, 214, 0), 2)
    side_c = 1.30
    y_row = top_c - 0.30 - side_c
    x = margin
    c.image(drawn, x, y_row, side_c, edge=ROLE["test"], lw=1.1)
    c.text(x + side_c / 2, y_row - 0.12, f"field, {len(cells)} whole white cells", size=6.4, weight="bold")
    c.text(x + side_c / 2, y_row - 0.28, "saturation Otsu + Azure-B prior", size=5.9, colour=MUTED)
    window = float(cfg["img_size"]) / float(inf["scale"])
    crop_side, crop_gap = 0.56, 0.08
    shown = min(len(cells), 4)
    x_crops = x + side_c + 0.42
    c.arrow(x + side_c + 0.06, y_row + side_c / 2, x_crops - 0.06, y_row + side_c / 2)
    for k in range(shown):
        cx, cy, _ = cells[k]
        crop = crop_window(field, cx, cy, window)
        row, col = divmod(k, 2)
        xk = x_crops + col * (crop_side + crop_gap)
        yk = y_row + side_c - (row + 1) * crop_side - row * crop_gap
        c.image(crop, xk, yk, crop_side, edge="#9A9A9A", lw=0.7)
        if probs is not None:
            # Dark on light: white text in a PDF reads as hidden text to plagiarism checkers
            c.text(xk + crop_side / 2, yk + 0.06, f"{probs[k]:.2f}", size=5.8, colour=INK, weight="bold",
                   va="bottom", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="#9A9A9A", lw=0.4, alpha=0.9))
    x_end_crops = x_crops + 2 * crop_side + crop_gap
    c.text((x_crops + x_end_crops) / 2, y_row - 0.12, f"crops of {window:.0f} px to 224 px", size=6.4,
           weight="bold")
    c.text((x_crops + x_end_crops) / 2, y_row - 0.28, "p(ALL) per cell, one model", size=5.9, colour=MUTED)
    x_m = x_end_crops + 0.42
    c.arrow(x_end_crops + 0.06, y_row + side_c / 2, x_m - 0.06, y_row + side_c / 2)
    box_w = 1.25
    c.box(x_m, y_row + 0.30, box_w, side_c - 0.60, "#E6E6E6", ec="#9A9A9A", lw=0.7)
    field_p = float(np.mean(probs)) if probs is not None and len(probs) else float("nan")
    c.text(x_m + box_w / 2, y_row + side_c / 2 + 0.12, "mean over cells", size=6.6, weight="bold")
    c.text(x_m + box_w / 2, y_row + side_c / 2 - 0.08, f"p(ALL) = {field_p:.2f}", size=6.6)
    x_d = x_m + box_w + 0.42
    c.arrow(x_m + box_w + 0.06, y_row + side_c / 2, x_d - 0.06, y_row + side_c / 2)
    dec_w = W - margin - x_d
    c.box(x_d, y_row + 0.30, dec_w, side_c - 0.60, FILL["test"], ec=ROLE["test"], lw=0.7)
    label = cfg["phase1"]["class_names"][int(field_p >= 0.5)]
    c.text(x_d + dec_w / 2, y_row + side_c / 2 + 0.12, f"predicted label: {label}", size=6.6, weight="bold")
    c.text(x_d + dec_w / 2, y_row + side_c / 2 - 0.08, "ALL iff p(ALL) $\\geq$ 0.5", size=6.2, colour=MUTED)
    return figstyle.finish(c.fig, out)[0]


def crop_window(img: np.ndarray, cx: float, cy: float, window: float, out: int = 224) -> np.ndarray:
    """A window of `window` px centred on (cx, cy), border-replicated, resized to `out` (as `crop_cells`)."""
    half = window / 2
    m = cv2.getAffineTransform(np.float32([[cx - half, cy - half], [cx + half, cy - half], [cx - half, cy + half]]),
                               np.float32([[0, 0], [out, 0], [0, out]]))
    return cv2.warpAffine(img, m, (out, out), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def example_field(cfg: Dict, n_cells: int = 4) -> tuple:
    """A representative Aria ALL field for the inference panel: of the fields with exactly `n_cells` scored cells,
    the one whose lowest seed-ensemble cell p(ALL) (MorphoMix, mean over every seed) is highest; returns its file
    name and the per-cell p(ALL) of one model, the primary-seed MorphoMix (each model scores the field alone)."""
    per_seed = []
    for seed in cfg["seeds"]:
        path = abs_path(os.path.join(cfg["results_dir"], "predictions", f"phase1_morpho_mix_seed{seed}_test_aria.json"))
        with open(path, encoding="utf-8") as f:
            pred = json.load(f)
        per_seed.append({os.path.basename(n): p for n, p, y in zip(pred["names"], pred["cell_probs"], pred["y_true"])
                         if y == 1 and len(p) == n_cells})
    common = set.intersection(*(set(d) for d in per_seed))
    ensemble = {n: np.mean([d[n] for d in per_seed], axis=0) for n in common}
    best = max(sorted(ensemble), key=lambda n: ensemble[n].min())
    return best, np.asarray(per_seed[cfg["seeds"].index(cfg["primary_seed"])][best])


COHORTS = [("val", "LeukemiaAttri (selection)"), ("allidb2", "ALL-IDB2 (single cells)"),
           ("aria", "Aria (whole fields)")]
ABLATION = {"b": "no $C_1$ (in-cell stain)", "c": "no $C_2$ (small cell)", "d": "no $C_3$ (background)",
            "e": "$C_1$ as RandStainNA"}  # arm f (no auxiliary cells) is in fig_aux


def table(cfg: Dict, name: str):
    path = abs_path(os.path.join(cfg["results_dir"], "tables", name))
    if not os.path.exists(path):
        print(f"  missing: {path}")
        return None
    if name.endswith(".csv"):
        return pd.read_csv(path)
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload if not isinstance(payload, dict) or "rows" not in payload else payload["rows"]


def arm_style(arm: str):
    ours = arm == "morpho_mix"
    return figstyle.ARM_COLOUR.get(arm, figstyle.COLOURS["grey"]), ours


def seed_strip(ax, y: float, values: List[float], colour: str, ours: bool, label_fmt: str = "{:.3f}",
               scale: float = 1.0):
    """Seeds as small hollow dots, the mean as a filled marker with a +/- sd bar, the mean printed beside it;
    `scale` enlarges markers and text together."""
    v = np.asarray(values, dtype=float)
    ax.scatter(v, np.full_like(v, y), s=13 * scale ** 2, facecolor="white", edgecolor=colour, linewidth=0.9 * scale,
               zorder=3)
    ax.errorbar(v.mean(), y, xerr=v.std(ddof=1) if len(v) > 1 else 0, fmt="o", ms=(6.2 if ours else 5.0) * scale,
                color=colour, mec="white", mew=0.7, elinewidth=1.4 * scale, capsize=0, zorder=4)
    ax.annotate(label_fmt.format(v.mean()), (v.mean(), y), xytext=(0, 6.5 * scale), textcoords="offset points",
                ha="center", fontsize=6.2 * scale, color=colour, fontweight="bold" if ours else "normal")


def arm_axis(ax, labels: List[str], bold_last: bool = True):
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ticks = ax.get_yticklabels()
    if bold_last and ticks:
        ticks[-1].set_fontweight("bold")
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#E3E3E3", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def figure_arms(cfg: Dict, present) -> List[str]:
    """The arms to draw, in reading order: the frozen foundation-model probes, the baselines, MorphoMix last."""
    order = [*FOUNDATION_ARMS, *[a for a in cfg["augmentations"] if a != "morpho_mix"], "morpho_mix"]
    return [a for a in order if a in set(present)]


def fig_main_results(cfg: Dict, out: str) -> Optional[str]:
    """ROC-AUC per arm and cohort (seeds and mean +/- sd), against the ROC-AUC that background colour statistics
    alone reach on the same cohort."""
    runs, bg = table(cfg, "extended_metrics_runs.json"), table(cfg, "shortcut_background_only_auc.json")
    if runs is None:
        return None
    rows = [{"augmentation": r["arm"], "seed": r["seed"], f"{r['cohort']}_roc_auc": r["roc_auc"]} for r in runs]
    arms = figure_arms(cfg, [r["augmentation"] for r in rows])
    labels = ["Background only"] * (bg is not None) + [figstyle.ARM_LABEL.get(a, a) for a in arms]
    offset = len(labels) - len(arms)
    fig, axes = plt.subplots(1, 3, figsize=(figstyle.WIDTH_DOUBLE, 0.55 + 0.40 * len(labels)), sharey=True,
                             sharex=True, layout="constrained")
    for ax, (key, title) in zip(axes, COHORTS):
        if bg is not None:  # label information in the background (a classifier fitted on the cohort), not a comparator
            ref = bg[key]
            ax.scatter([ref], [0], marker="D", s=26, color=BG_REF, zorder=4)
            ax.annotate(f"{ref:.3f}", (ref, 0), xytext=(0, 6.5), textcoords="offset points", ha="center",
                        fontsize=6.2, color=BG_REF)
            ax.axvline(ref, color=BG_REF, linestyle=(0, (2, 3)), linewidth=0.7, alpha=0.6, zorder=1)
        for i, arm in enumerate(arms):
            colour, ours = arm_style(arm)
            seed_strip(ax, i + offset, [r[f"{key}_roc_auc"] for r in rows
                                        if r["augmentation"] == arm and f"{key}_roc_auc" in r], colour, ours)
        n_fm = sum(a in FOUNDATION_ARMS for a in arms)
        for y in (offset - 0.5, offset + n_fm - 0.5)[:1 + bool(n_fm)]:
            ax.axhline(y, color="#CCCCCC", linewidth=0.6)
        ax.set_title(title, fontsize=7.6, fontweight="bold", loc="left")
        ax.set_xlabel("ROC-AUC")
        arm_axis(ax, labels)
    axes[0].set_xlim(0.35, 1.0)
    return figstyle.finish(fig, out)[0]


SHORTCUT_PARTS = ("residual", "swap")
SHORTCUT_SCALE = 1.3  # text and markers of the shortcut figures, against the other result figures: dense panels


def fig_shortcut(cfg: Dict, out: str, parts: tuple = SHORTCUT_PARTS) -> Optional[str]:
    """`residual` (two panels): ROC-AUC before and after removing what background statistics explain of each score
    (residual AUC), with the background-only reference; `swap` (two panels, ALL-IDB2): ROC-AUC with the cell on a
    same-class minus on an other-class background (a class-specific background effect), for every trained arm with
    and without the auxiliary cells, and single-view ROC-AUC on the original image and with the background removed
    (cell on black). Panels are lettered in order over the parts drawn."""
    res, swap = table(cfg, "shortcut_residual_runs.csv"), table(cfg, "shortcut_bg_swap_runs.csv")
    bg = table(cfg, "shortcut_background_only_auc.json")
    if res is None or swap is None or "class_specific_effect" not in swap:
        return None
    k = SHORTCUT_SCALE
    arms = figure_arms(cfg, res["arm"])
    trained = [a for a in arms if a not in FOUNDATION_ARMS]
    rows = [(a, a, figstyle.ARM_LABEL.get(a, a)) for a in trained]
    rows += [(f"abl{NO_AUX[a]}", a, f"{figstyle.ARM_LABEL.get(a, a)}, no aux") for a in trained if a in NO_AUX]
    n_rows = {"residual": len(arms), "swap": len(rows)}
    letters = iter("abcd")
    with plt.rc_context({"font.size": 8 * k}):
        height = 0.8 + 0.85 * len(parts) + 0.34 * sum(n_rows[p] for p in parts)
        fig = plt.figure(figsize=(figstyle.WIDTH_DOUBLE, height), layout="constrained")
        grid = fig.add_gridspec(len(parts), 2, height_ratios=[n_rows[p] + 1.2 for p in parts])
        for r, part in enumerate(parts):
            left = fig.add_subplot(grid[r, 0])
            panels = (left, fig.add_subplot(grid[r, 1], sharey=left))
            panels[1].tick_params(labelleft=False)
            if part == "residual":
                draw_residual(panels, res, bg, arms, letters)
            else:
                draw_swap(panels, swap[swap["cohort"] == "allidb2"], rows, len(trained), letters)
        grey = figstyle.COLOURS["grey"]
        handles = {
            "residual": [plt.Line2D([], [], marker="o", linestyle="", color=grey, markersize=4.5 * k,
                                    label="model ROC-AUC"),
                         plt.Line2D([], [], marker="s", linestyle="", markerfacecolor="white", markeredgecolor=grey,
                                    markersize=4.5 * k, label="residual ROC-AUC"),
                         plt.Line2D([], [], color=BG_REF, linestyle=(0, (2, 3)), label="background only")],
            "swap": [plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=grey,
                                markersize=3.5 * k, label="seed")],
        }
        legend = [h for p in parts for h in handles[p]]
        fig.legend(handles=legend, loc="outside lower left", ncol=len(legend), fontsize=6.4 * k)
        return figstyle.finish(fig, out)[0]


def draw_residual(axes, res: pd.DataFrame, bg: Optional[Dict], arms: List[str], letters) -> None:
    """Model and residual ROC-AUC per arm on ALL-IDB2 and Aria (mean of seeds)."""
    k = SHORTCUT_SCALE
    for ax, key, title in zip(axes, ("allidb2", "aria"), ("ALL-IDB2, residual", "Aria, residual")):
        sub = res[(res["cohort"] == key) & res["arm"].isin(arms)].groupby("arm")[["auc", "auc_residual"]].mean()
        for i, arm in enumerate(arms):
            colour, ours = arm_style(arm)
            raw, resid = sub.loc[arm, "auc"], sub.loc[arm, "auc_residual"]
            ax.plot([resid, raw], [i, i], color=colour, linewidth=(2.6 if ours else 1.8) * k, alpha=0.55,
                    solid_capstyle="round", zorder=2)
            ax.scatter([raw], [i], s=30 * k ** 2, color=colour, edgecolor="white", linewidth=0.7, zorder=3)
            ax.scatter([resid], [i], s=30 * k ** 2, marker="s", facecolor="white", edgecolor=colour, linewidth=1.3,
                       zorder=3)
            ax.annotate(f"{resid:.3f}", (resid, i), xytext=(-6, 0), textcoords="offset points", ha="right",
                        va="center", fontsize=6.0 * k, color=colour, fontweight="bold" if ours else "normal")
        if bg is not None:
            ax.axvline(bg[key], color=BG_REF, linestyle=(0, (2, 3)), linewidth=0.9, alpha=0.7, zorder=1)
        ax.set_title(f"({next(letters)}) {title}", fontsize=7.6 * k, fontweight="bold", loc="left")
        ax.set_xlabel("ROC-AUC")
        ax.set_xlim(0.45, 1.0)
        arm_axis(ax, [figstyle.ARM_LABEL.get(a, a) for a in arms])


def draw_swap(axes, swap: pd.DataFrame, rows: List[tuple], n_trained: int, letters) -> None:
    """ALL-IDB2 background swap (class-specific effect per seed) and background removal (mean of seeds)."""
    k = SHORTCUT_SCALE
    for ax, cols, title, xlabel in ((axes[0], ("class_specific_effect",), "ALL-IDB2, background swap",
                                     "ROC-AUC, same-class minus\nother-class background"),
                                    (axes[1], ("auc", "auc_none"), "ALL-IDB2, background removed",
                                     "single-view ROC-AUC")):
        for i, (arm, style_arm, _) in enumerate(rows):
            colour, ours = arm_style(style_arm)
            sub = swap[swap["arm"] == arm]
            if sub.empty:
                continue
            if cols[0] == "class_specific_effect":
                seed_strip(ax, i, sub[cols[0]].tolist(), colour, ours, label_fmt="{:+.3f}", scale=k)
            else:
                before, after = sub[cols[0]].mean(), sub[cols[1]].mean()
                if abs(after - before) >= 0.02:  # a shorter arrow is all head: the printed value says enough
                    arrow = dict(arrowstyle="-|>", color=colour, lw=1.4 * k, alpha=0.7, shrinkA=3, shrinkB=2,
                                 mutation_scale=9 * k)
                    ax.annotate("", (after, i), (before, i), arrowprops=arrow)
                ax.scatter([before], [i], s=26 * k ** 2, color=colour, edgecolor="white", linewidth=0.7, zorder=3)
                ax.annotate(f"{after:.3f}", (after, i), xytext=(0, 6.5 * k), textcoords="offset points", ha="center",
                            fontsize=6.0 * k, color=colour, fontweight="bold" if ours else "normal")
        if cols[0] == "class_specific_effect":
            ax.axvline(0, color="#333333", linewidth=0.8)
            ax.margins(x=0.2)
        else:
            ax.set_xlim(0.55, 0.95)
        ax.axhline(n_trained - 0.5, color="#CCCCCC", linewidth=0.6)
        ax.set_title(f"({next(letters)}) {title}", fontsize=7.6 * k, fontweight="bold", loc="left")
        ax.set_xlabel(xlabel)
        arm_axis(ax, [label for *_, label in rows], bold_last=False)


def fig_ablation(cfg: Dict, out: str) -> Optional[str]:
    """Full MorphoMix minus the ablation arm in ROC-AUC per cohort, paired by seed (seeds and mean +/- sd), the sign
    of the supplement table: positive = the component helps."""
    runs = table(cfg, "component_ablation_runs.json")
    if runs is None:
        return None
    full = {r["seed"]: r for r in runs if r["arm"] == "a"}
    order = [k for k in ABLATION if any(r["arm"] == k for r in runs)]
    if not full or not order:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(figstyle.WIDTH_DOUBLE, 0.6 + 0.40 * len(order)), sharey=True,
                             sharex=True, layout="constrained")
    colour = figstyle.COLOURS["darkorange"]
    for ax, (key, title) in zip(axes, COHORTS):
        for i, k in enumerate(order):
            deltas = [full[r["seed"]][f"{key}_roc_auc"] - r[f"{key}_roc_auc"]
                      for r in runs if r["arm"] == k and r["seed"] in full]
            seed_strip(ax, i, deltas, colour, False, label_fmt="{:+.3f}")
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_title(title, fontsize=7.6, fontweight="bold", loc="left")
        ax.margins(x=0.15)
        arm_axis(ax, [ABLATION[k] for k in order], bold_last=False)
    seeds = sorted({r["seed"] for r in runs if r["arm"] in order})
    fig.supxlabel(f"ROC-AUC, full MorphoMix minus ablated (paired by seed, seeds {seeds[0]}-{seeds[-1]})",
                  fontsize=7.5)
    return figstyle.finish(fig, out)[0]


CAM_STAGES = (3, 4)  # stage 3 is MorphoMix's best and stage 4 its worst localisation stage: both are shown


def cam_examples(cfg: Dict, arms: List[str], per_class: int = 2) -> List[tuple]:
    """Representative ALL-IDB2 cells chosen without looking at any CAM: per class, the cells that every arm's seed
    ensemble (mean p over its seeds) classifies most confidently, one per cell group, with a whole central cell."""
    seen, confidence = {}, {}
    for arm in arms:
        for seed in cfg["seeds"]:
            name = f"phase1_{arm}_seed{seed}_test_allidb2.json"
            path = abs_path(os.path.join(cfg["results_dir"], "predictions", name))
            with open(path, encoding="utf-8") as f:
                pred = json.load(f)
            for n, y, p in zip(pred["names"], pred["y_true"], pred["probs"]):
                seen[os.path.basename(n)] = y
                confidence.setdefault((os.path.basename(n), arm), []).append(p[y])
    groups = pd.read_csv(abs_path("data/processed/splits/allidb2_groups.csv")).set_index("file")["group"]
    tau = float(cfg["morpho_threshold"])
    picked = []
    for y in (0, 1):
        ranked = sorted((n for n in seen if seen[n] == y),
                        key=lambda n: -min(np.mean(confidence[(n, a)]) for a in arms))
        used = set()
        for name in ranked:
            img = rgb(f"data/processed/test_allidb2/{cfg['phase1']['class_names'][y]}/{name}")
            mask = central_cell_mask(img, tau) > 0
            frame = mask[:6].any() or mask[-6:].any() or mask[:, :6].any() or mask[:, -6:].any()
            if groups[name] in used or frame or not 0.08 < mask.mean() < 0.45:
                continue
            used.add(groups[name])
            picked.append((name, y, img, mask))
            if len(used) == per_class:
                break
    return picked


def fig_cam(cfg: Dict, out: str) -> Optional[str]:
    """Layer-CAM (true class) at every CAM_STAGES stage of every arm's primary-seed model on one representative
    Normal and one ALL cell of ALL-IDB2, with the pseudo ground-truth cell outline."""
    import torch
    from scripts.evaluate import N_STAGES, load_phase1_model
    from src.augmentations.transforms import get_val_transforms
    from src.cam.layer_cam import LayerCAM
    from src.models.factory import get_target_cam_layer

    arms = [a for a in cfg["augmentations"] if a != "morpho_mix"] + ["morpho_mix"]
    examples = cam_examples(cfg, arms, per_class=1)
    tf = get_val_transforms(cfg["img_size"])
    x = torch.stack([tf(Image.fromarray(img)) for _, _, img, _ in examples])
    y = torch.tensor([label for _, label, _, _ in examples])
    cams = {}
    for arm in arms:
        ckpt = f"phase1_{arm}_seed{cfg['primary_seed']}_best.pt"
        model = load_phase1_model(cfg, abs_path(os.path.join(cfg["results_dir"], "checkpoints", ckpt)),
                                  torch.device("cpu"))
        for stage in CAM_STAGES:
            cam = LayerCAM(model, get_target_cam_layer(model, stage_idx=stage - N_STAGES - 1))
            cams[(arm, stage)] = cam.generate(x, y)[:, 0].numpy()
            cam.remove_hooks()

    cmap = plt.get_cmap("inferno")
    rows = [(e, stage) for e in range(len(examples)) for stage in CAM_STAGES]
    n_cols = len(arms) + 1
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(figstyle.WIDTH_DOUBLE, figstyle.WIDTH_DOUBLE / n_cols
                                                         * len(rows) + 0.3), gridspec_kw=dict(wspace=0.04, hspace=0.04))
    for r, (e, stage) in enumerate(rows):
        _, label, img, mask = examples[e]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c, arm in enumerate([None] + arms):
            ax = axes[r, c]
            if arm is None:
                shown = img.copy()
                cv2.drawContours(shown, contours, -1, (60, 220, 90), 2)
            else:
                heat = cams[(arm, stage)][e]
                weight = 0.85 * heat[..., None]  # the cell stays visible where the map is low
                shown = ((1 - weight) * img + weight * cmap(heat)[..., :3] * 255).astype(np.uint8)
                cv2.drawContours(shown, contours, -1, (255, 255, 255), 1)
            ax.imshow(shown, interpolation="lanczos")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(figstyle.COLOURS["orange"] if arm == "morpho_mix" else "#BBBBBB")
                spine.set_linewidth(1.6 if arm == "morpho_mix" else 0.6)
            if r == 0:
                title = figstyle.ARM_LABEL.get(arm, arm) if arm else "Cell"
                ax.set_title(title, fontsize=7.2, fontweight="bold" if arm == "morpho_mix" else "normal", pad=4)
            if c == 0:
                ax.set_ylabel(f"{cfg['phase1']['class_names'][label]}, stage {stage}", fontsize=7.0)
    return figstyle.finish(fig, out)[0]


NO_AUX = {"basic": "h", "hed_jitter": "i", "randstainna": "j", "stain_mixup": "k", "morpho_mix": "f"}


def fig_aux(cfg: Dict, out: str) -> Optional[str]:
    """ROC-AUC of every arm without (hollow) and with (filled) the auxiliary cells on the ablation seeds, and
    MorphoMix with the auxiliary cells colour-matched to the C-NMC template (control arm g, square)."""
    runs, abl = table(cfg, "extended_metrics_runs.json"), table(cfg, "component_ablation_runs.json")
    if runs is None or abl is None:
        return None
    seeds = set(cfg["ablation_seeds"])
    arms = [a for a in figure_arms(cfg, NO_AUX) if any(r["arm"] == NO_AUX[a] for r in abl)]
    if not arms:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(figstyle.WIDTH_DOUBLE, 0.7 + 0.40 * len(arms)), sharey=True,
                             sharex=True, layout="constrained")
    for ax, (key, title) in zip(axes, COHORTS):
        for i, arm in enumerate(arms):
            colour, ours = arm_style(arm)
            with_aux = np.mean([r["roc_auc"] for r in runs
                                if r["arm"] == arm and r["cohort"] == key and r["seed"] in seeds])
            without = np.mean([r[f"{key}_roc_auc"] for r in abl if r["arm"] == NO_AUX[arm]])
            ax.annotate("", xy=(with_aux, i), xytext=(without, i),
                        arrowprops=dict(arrowstyle="-|>", color=colour, lw=2.2 if ours else 1.4, alpha=0.6,
                                        shrinkA=4, shrinkB=4, mutation_scale=8))
            ax.scatter([without], [i], s=30, facecolor="white", edgecolor=colour, linewidth=1.3, zorder=3)
            ax.scatter([with_aux], [i], s=34 if ours else 28, color=colour, edgecolor="white", linewidth=0.7, zorder=3)
            if ours and any(r["arm"] == "g" for r in abl):
                matched = np.mean([r[f"{key}_roc_auc"] for r in abl if r["arm"] == "g"])
                ax.scatter([matched], [i], marker="s", s=26, color=figstyle.COLOURS["darkorange"], zorder=4)
        ax.set_title(title, fontsize=7.6, fontweight="bold", loc="left")
        ax.set_xlabel(f"ROC-AUC, mean of seeds {min(seeds)}-{max(seeds)}")
        ax.margins(x=0.08)  # sharex: one x range for every panel, so changes compare across cohorts
        arm_axis(ax, [figstyle.ARM_LABEL.get(a, a) for a in arms])
    grey = figstyle.COLOURS["grey"]
    handles = [plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=grey,
                          markersize=4.5, label="without auxiliary cells"),
               plt.Line2D([], [], marker="o", linestyle="", color=grey, markersize=4.5, label="with auxiliary cells"),
               plt.Line2D([], [], marker="s", linestyle="", color=figstyle.COLOURS["darkorange"], markersize=4.5,
                          label="colour-matched auxiliary cells")]
    fig.legend(handles=handles, loc="outside lower left", ncol=3, fontsize=6.4)
    return figstyle.finish(fig, out)[0]


def fig_val_curves(cfg: Dict, out: str) -> Optional[str]:
    """Val ROC-AUC per epoch, mean +/- sd over seeds: the whole curve, not only the selected checkpoint; below it, one
    tick row per arm marks the epoch each run selected."""
    path = abs_path(os.path.join(cfg["results_dir"], "logs", "training_log.csv"))
    if not os.path.exists(path):
        return None
    log = pd.read_csv(path)
    main_ids = "phase1_" + log["augmentation"] + "_seed" + log["seed"].astype(str)
    log = log[log["experiment_id"] == main_ids].drop_duplicates(["experiment_id", "epoch"], keep="last")
    arms = [a for a in cfg["augmentations"] if not log[log["augmentation"] == a].empty]
    fig, (ax, sel) = plt.subplots(2, 1, figsize=(figstyle.WIDTH_SINGLE, 3.3), sharex=True, layout="constrained",
                                  height_ratios=[3, 0.22 * len(arms) + 0.2])
    for i, arm in enumerate(arms):
        sub = log[log["augmentation"] == arm]
        colour, ours = arm_style(arm)
        curve = sub.groupby("epoch")["val_roc_auc"].agg(["mean", "std"])
        ax.plot(curve.index, curve["mean"], color=colour, linewidth=2.6 if ours else 1.1,
                label=figstyle.ARM_LABEL.get(arm, arm), zorder=3 if ours else 2)
        ax.fill_between(curve.index, curve["mean"] - curve["std"], curve["mean"] + curve["std"], color=colour,
                        alpha=0.15 if ours else 0.07, linewidth=0)
        picked = sub.loc[sub.groupby("experiment_id")["val_roc_auc"].idxmax(), "epoch"]
        sel.scatter(picked, np.full(len(picked), i), marker="|", s=45, color=colour, linewidth=1.4, zorder=4)
    ax.set_ylabel("selection-cohort ROC-AUC")
    sel.set_xlabel("epoch")
    sel.set_title("selected epoch per seed", fontsize=7.0, loc="left", pad=2)
    arm_axis(sel, [figstyle.ARM_LABEL.get(a, a) for a in arms], bold_last=False)
    sel.tick_params(axis="y", labelsize=6.8, length=0)
    ax.grid(color="#E3E3E3", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(color=figstyle.COLOURS["lightgrey"], alpha=0.35, linewidth=0))
    labels.append("$\\pm$1 sd")
    ax.legend(handles, labels, fontsize=7.2, ncol=2, loc="lower left", handlelength=1.6, columnspacing=1.0)
    return figstyle.finish(fig, out)[0]


def mean_auc(runs: List[Dict], arm: str, cohort: str) -> List[float]:
    return [r["roc_auc"] for r in runs if r["arm"] == arm and r["cohort"] == cohort]


def fig_graphical_abstract(cfg: Dict, out: str) -> Optional[str]:
    """Elsevier graphical abstract (13.28 x 5.31 in at 300 dpi, readable at 5 x 13 cm): the cross-laboratory design,
    the background-only reference on every cohort, and every method on the two background-neutral cohorts (seed
    means, +/- sd; no method highlighted)."""
    runs, bg = table(cfg, "extended_metrics_runs.json"), table(cfg, "shortcut_background_only_auc.json")
    rep, rep_bg = table(cfg, "confirm_l100x_runs.csv"), table(cfg, "confirm_l100x_background.json")
    if runs is None or bg is None or rep is None or rep_bg is None:
        return None
    W, H = 13.28, 5.31
    c = Canvas(W, H)
    title_y = H - 0.42

    # (a) one laboratory per role
    c.box(0.12, 0.12, 4.02, H - 0.24, "#F6F6F6")
    c.text(0.36, title_y, "Cross-laboratory design", size=19, weight="bold", ha="left")
    design = [("train", "Train", [("cnmc", "C-NMC", "cells"), ("bodzas", "Bodzas", "auxiliary")]),
              ("select", "Select", [("leukemiaattri", "LeukemiaAttri", "cells")]),
              ("test", "Test", [("allidb2", "ALL-IDB2", "cells"), ("aria", "Aria", "whole fields")])]
    side, pitch = 0.92, 1.30
    for r, (role, label, cohorts) in enumerate(design):
        y = title_y - 0.38 - side - r * 1.42
        c.text(0.36, y + side / 2, label, size=19, weight="bold", colour=ROLE[role], ha="left")
        if r:
            c.arrow(0.74, y + side + 0.42, 0.74, y + side / 2 + 0.24, colour=ROLE[role], lw=2.0, head=18)
        for k, (key, name, kind) in enumerate(cohorts):
            x = 1.62 + k * pitch
            img = rgb(f"data/processed/test_aria/ALL/{example_field(cfg)[0]}" if key == "aria" else EXAMPLES[key])
            c.image(img, x, y, side, edge=ROLE[role], lw=1.6)
            c.text(x + side / 2, y - 0.03, name, size=15, weight="bold", va="top")
            c.text(x + side / 2, y - 0.27, kind, size=13, colour=MUTED, va="top")

    # (b) the background-only reference: grouped cross-validated ROC-AUC of background statistics within each cohort
    x0, w = 4.30, 4.30
    c.box(x0, 0.12, w, H - 0.24, "#F6F6F6")
    c.text(x0 + 0.24, title_y, "Background alone", size=19, weight="bold", ha="left")
    c.text(x0 + 0.24, title_y - 0.42, "predicts the test labels", size=16, ha="left", colour=MUTED)
    cohorts = [("Selection", bg["val"], "select"), ("Replicate", rep_bg["background_only_roc_auc"], "select"),
               ("ALL-IDB2", bg["allidb2"], "test"), ("Aria", bg["aria"], "test")]
    ax_x, ax_w, y_ax, h_ax = x0 + 1.55, 2.20, 0.75, 3.30
    ax = c.fig.add_axes([ax_x / W, y_ax / H, ax_w / W, h_ax / H])
    for i, (name, value, role) in enumerate(cohorts):
        ax.barh(i, value, height=0.58, color=ROLE[role], alpha=0.85, zorder=3)
        ax.text(-0.05, i, name, fontsize=15, ha="right", va="center", fontweight="bold", color=ROLE[role])
        ax.text(value + 0.035, i, f"{value:.2f}", fontsize=15, ha="left", va="center", color=ROLE[role], zorder=6,
                bbox=dict(boxstyle="square,pad=0.1", fc="#F6F6F6", ec="none"))
    ax.axvline(0.5, color=MUTED, linewidth=1.4, linestyle="--", zorder=4)
    ax.text(0.5, -0.75, "chance", fontsize=12, color=MUTED, ha="center", va="bottom")
    ax.set_ylim(len(cohorts) - 0.5, -0.9)
    ax.set_yticks([])
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=12, length=3)
    ax.set_xlabel("ROC-AUC", fontsize=13)
    ax.patch.set_alpha(0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    # (c) every method on the two background-neutral cohorts, seed mean +/- sd, none highlighted
    x0c = 8.76
    c.box(x0c, 0.12, W - 0.12 - x0c, H - 0.24, "#F6F6F6")
    c.text(x0c + 0.24, title_y, "Background-neutral cohorts", size=18, weight="bold", ha="left")
    c.text(x0c + 0.24, title_y - 0.42, "MorphoMix vs each comparator: n.s.", size=15, ha="left", colour=MUTED)
    c.text(x0c + 0.24, title_y - 0.72, "primary analysis; mean $\\pm$ sd over seeds", size=12, ha="left", colour=MUTED)
    arms = list(cfg["augmentations"]) + ["dinobloom_s"]
    sel = {a: mean_auc(runs, a, "val") for a in arms}
    rep_auc = {a: rep.loc[rep["arm"] == a, "roc_auc"].tolist() for a in arms}
    ax_x, ax_w = 10.90, 1.70
    for k, (title, values) in enumerate([("Selection", sel), ("Replicate", rep_auc)]):
        y_ax, h_ax = 2.48 - k * 2.08, 1.62
        ax = c.fig.add_axes([ax_x / W, y_ax / H, ax_w / W, h_ax / H])
        for i, arm in enumerate(arms):
            v = np.asarray(values[arm], dtype=float)
            colour = figstyle.ARM_COLOUR[arm]
            ax.errorbar(v.mean(), i, xerr=v.std(ddof=1) if len(v) > 1 else 0, fmt="o", ms=8, color=colour,
                        elinewidth=2.2, capsize=0, zorder=4)
            y_row = y_ax + h_ax * (len(arms) - 0.5 - i) / len(arms)
            c.text(ax_x - 0.12, y_row, figstyle.ARM_LABEL[arm].replace(" probe", ""), size=12.5, ha="right")
        ax.set_ylim(len(arms) - 0.5, -0.5)
        ax.set_yticks([])
        ax.set_xlim(0.60, 0.85)
        ax.set_xticks([0.6, 0.7, 0.8])
        ax.tick_params(labelsize=12, labelbottom=bool(k), length=3)
        ax.grid(axis="x", color="#DCDCDC", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.patch.set_alpha(0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        c.text(ax_x + ax_w + 0.10, y_ax + h_ax / 2, title, size=14, weight="bold", ha="left", rotation=90,
               colour=ROLE["select"])
    with plt.rc_context({"savefig.bbox": "standard"}):  # the exact Elsevier size, not a tight crop
        return figstyle.finish(c.fig, out)[0]


# C-NMC training cells for fig_c1_examples, chosen by eye: typical transfers, then the pale-patch failure, where the
# Otsu saturation split puts "cytoplasm" inside a cell with little visible cytoplasm
C1_EXAMPLES = [
    ("typical", "data/processed/train/ALL/CNMC_fold_0_UID_11_11_3_all.png"),
    ("typical", "data/processed/train/Normal/CNMC_fold_2_UID_H18_10_1_hem.png"),
    ("typical", "data/processed/train/ALL/CNMC_fold_1_UID_16_30_2_all.png"),
    ("typical", "data/processed/train/Normal/CNMC_fold_0_UID_H11_12_1_hem.png"),
    ("pale patches", "data/processed/train/ALL/CNMC_fold_0_UID_35_11_8_all.png"),
    ("pale patches", "data/processed/train/Normal/CNMC_fold_0_UID_H24_25_3_hem.png"),
]
C1_REF_PERCENTILE = 3.0  # references: the bank rows nearest both tails of the first two principal axes of the Lab means
CYTOPLASM = (0, 200, 220)


def fig_c1_examples(cfg: Dict, out: str) -> Optional[str]:
    """MorphoMix C1 at full strength on C-NMC training cells (rows) towards MLL23 bank references (columns), with
    the cytoplasm the Otsu split assigns and each reference's nucleus / cytoplasm mean colour."""
    import torch
    from src.augmentations.stain import lab_to_rgb, match_regions, split_nucleus
    from src.augmentations.style_bank import StyleBank

    bank = StyleBank.load(cfg["style_bank"]["path"])
    means = bank.mean.flatten(1).double()
    scores = (means - means.mean(dim=0)) @ torch.linalg.svd(means - means.mean(dim=0), full_matrices=False)[2][:2].T
    lo, hi = (torch.quantile(scores, q / 100, dim=0) for q in (C1_REF_PERCENTILE, 100 - C1_REF_PERCENTILE))
    targets = [(lo[0], 0.0), (hi[0], 0.0), (0.0, lo[1]), (0.0, hi[1])]
    refs = [int((scores - torch.tensor(t, dtype=scores.dtype)).norm(dim=1).argmin()) for t in targets]
    ref_mean, ref_sd = bank.mean[refs], bank.sd[refs]
    swatch = (lab_to_rgb(ref_mean.permute(0, 2, 1)[..., None]).squeeze(-1).permute(0, 2, 1).numpy() * 255)
    tau = float(cfg["morpho_threshold"])

    n_cols = 2 + len(refs)
    side, gap, label_w = 0.98, 0.06, 0.95
    head = 0.78
    W = label_w + n_cols * side + (n_cols - 1) * gap + 0.08
    H = head + len(C1_EXAMPLES) * (side + gap) + 0.10
    c = Canvas(W, H)
    x_col = [label_w + j * (side + gap) for j in range(n_cols)]
    y_head = H - head
    c.text(x_col[0] + side / 2, y_head + 0.14, "C-NMC cell", size=7.5, weight="bold")
    c.text(x_col[1] + side / 2, y_head + 0.14, "cytoplasm", size=7.5, weight="bold", colour="#008A99")
    x_ref0, x_ref1 = x_col[2], x_col[-1] + side
    c.ax.plot([x_ref0, x_ref1], [H - 0.20] * 2, color=figstyle.COLOURS["orange"], lw=1.6, solid_capstyle="butt")
    c.text((x_ref0 + x_ref1) / 2, H - 0.10, "$C_1$ towards MLL23 references", size=7.5, weight="bold",
           colour=figstyle.COLOURS["darkorange"])
    for j, sw in enumerate(swatch):
        xc, yc = x_col[2 + j] + side / 2, y_head + 0.30
        c.ax.add_patch(FancyBboxPatch((xc - 0.17, yc - 0.17), 0.34, 0.34, boxstyle="round,pad=0,rounding_size=0.08",
                                      fc=sw[1] / 255, ec="#9A9A9A", lw=0.5, zorder=3))
        c.ax.add_patch(plt.Circle((xc, yc), 0.11, fc=sw[0] / 255, ec="none", zorder=4))
        c.text(xc + 0.24, yc, f"{j + 1}", size=7.0, colour=MUTED, ha="left")

    groups: Dict[str, List[float]] = {}
    for r, (group, path) in enumerate(C1_EXAMPLES):
        img = rgb(path)
        x = torch.from_numpy(img.copy()).permute(2, 0, 1)[None].float() / 255
        cell = torch.from_numpy(cell_mask(img, tau))[None, None]
        nucleus = split_nucleus(x, cell)[0, 0].numpy()
        split = img.copy()
        split[(cell[0, 0].numpy() > 0) & ~nucleus] = CYTOPLASM
        matched = match_regions(x.expand(len(refs), -1, -1, -1), cell.expand(len(refs), -1, -1, -1), ref_mean, ref_sd)
        renders = [np.uint8(np.round(255 * m.permute(1, 2, 0).numpy())) for m in matched]
        rows, cols = np.nonzero(cell[0, 0].numpy())
        half = min(int(max(np.ptp(rows), np.ptp(cols)) / 2) + 10, img.shape[0] // 2)  # C-NMC crops are mostly black
        r0 = int(np.clip((rows.min() + rows.max()) // 2 - half, 0, img.shape[0] - 2 * half))
        c0 = int(np.clip((cols.min() + cols.max()) // 2 - half, 0, img.shape[1] - 2 * half))
        y = y_head - (r + 1) * (side + gap) + gap
        for j, shown in enumerate([img, split, *renders]):
            edge, lw = (figstyle.COLOURS["orange"], 0.9) if j >= 2 else ("#BBBBBB", 0.6)
            c.image(shown[r0:r0 + 2 * half, c0:c0 + 2 * half], x_col[j], y, side, edge=edge, lw=lw)
        label = "ALL" if "/ALL/" in path else "Normal"
        c.text(label_w - 0.08, y + side / 2, label, size=7.0, ha="right", colour=MUTED)
        groups.setdefault(group, []).extend([y, y + side])
    for group, ys in groups.items():
        y0, y1 = min(ys), max(ys)
        colour = figstyle.COLOURS["grey"] if group == "typical" else BG_REF
        c.ax.plot([0.30, 0.30], [y0 + 0.04, y1 - 0.04], color=colour, lw=1.6, solid_capstyle="butt")
        c.text(0.18, (y0 + y1) / 2, group, size=7.5, weight="bold", colour=colour, rotation=90)
    return figstyle.finish(c.fig, out)[0]


def ensemble_probs(cfg: Dict, arm: str, cohort: str):
    """Seed-ensemble p(ALL) (mean over every seed, aligned by image name) and the labels of one arm and cohort."""
    probs, labels = {}, {}
    for seed in cfg["seeds"]:
        path = abs_path(os.path.join(cfg["results_dir"], "predictions", f"phase1_{arm}_seed{seed}_{cohort}.json"))
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            pred = json.load(f)
        for n, p, y in zip(pred["names"], pred["probs"], pred["y_true"]):
            probs.setdefault(n, []).append(p[1])
            labels[n] = y
    names = sorted(n for n in probs if len(probs[n]) == len(cfg["seeds"]))
    return np.array([np.mean(probs[n]) for n in names]), np.array([labels[n] for n in names])


def fig_reliability(cfg: Dict, out: str, n_bins: int = 10) -> Optional[str]:
    """Reliability diagrams of every main arm's seed ensemble per cohort: observed ALL fraction against mean p(ALL)
    in equal-width bins (marker area ~ images in the bin), with the images per bin below."""
    arms = [a for a in cfg["augmentations"] if a != "morpho_mix"] + ["morpho_mix"]
    cohorts = [("val", COHORTS[0][1]), ("test_allidb2", COHORTS[1][1]), ("test_aria", COHORTS[2][1])]
    data = {(a, k): ensemble_probs(cfg, a, k) for a in arms for k, _ in cohorts}
    if any(v is None for v in data.values()):
        return None
    fig = plt.figure(figsize=(figstyle.WIDTH_DOUBLE, 3.2), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[3.2, 1])
    edges = np.linspace(0, 1, n_bins + 1)
    width = 1 / n_bins / (len(arms) + 1)
    for col, (key, title) in enumerate(cohorts):
        ax = fig.add_subplot(grid[0, col])
        hist = fig.add_subplot(grid[1, col], sharex=ax)
        ax.plot([0, 1], [0, 1], color="#333333", linestyle=(0, (3, 3)), linewidth=0.8, zorder=1)
        for i, arm in enumerate(arms):
            p, y = data[(arm, key)]
            colour, ours = arm_style(arm)
            idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
            n = np.bincount(idx, minlength=n_bins)
            seen = n > 0
            conf = np.bincount(idx, weights=p, minlength=n_bins)[seen] / n[seen]
            frac = np.bincount(idx, weights=y, minlength=n_bins)[seen] / n[seen]
            ax.plot(conf, frac, color=colour, linewidth=1.8 if ours else 1.0, alpha=0.9, zorder=3 if ours else 2,
                    label=figstyle.ARM_LABEL.get(arm, arm))
            ax.scatter(conf, frac, s=4 + 60 * np.sqrt(n[seen] / len(p)), color=colour, edgecolor="white",
                       linewidth=0.4, zorder=4 if ours else 3)
            hist.bar(edges[:-1] + (i + 1) * width, n, width=width, color=colour, align="center", linewidth=0)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_title(title, fontsize=7.6, fontweight="bold", loc="left")
        ax.tick_params(labelbottom=False)
        hist.set_yscale("log")
        hist.set_xlabel("mean p(ALL) in bin")
        for a in (ax, hist):
            a.grid(color="#E3E3E3", linewidth=0.6)
            a.set_axisbelow(True)
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
        if col == 0:
            ax.set_ylabel("observed fraction ALL")
            hist.set_ylabel("images")
    handles, labels = fig.axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="#333333", linestyle=(0, (3, 3)), linewidth=0.8))
    labels.append("perfect calibration")
    fig.legend(handles, labels, loc="outside lower center", ncol=len(arms) + 1, fontsize=6.4)
    return figstyle.finish(fig, out)[0]


FIGURES: Dict[str, Callable[[Dict, str], Optional[str]]] = {
    "fig_overview": fig_overview,
    "fig_main_results": fig_main_results,
    "fig_shortcut": fig_shortcut,
    "fig_shortcut_residual": partial(fig_shortcut, parts=("residual",)),
    "fig_shortcut_swap": partial(fig_shortcut, parts=("swap",)),
    "fig_ablation": fig_ablation,
    "fig_aux": fig_aux,
    "fig_cam": fig_cam,
    "fig_val_curves": fig_val_curves,
    "fig_graphical_abstract": fig_graphical_abstract,
    "fig_c1_examples": fig_c1_examples,
    "fig_reliability": fig_reliability,
}


def paperfigs_main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Paper figures from results_dir/tables, predictions and data.")
    ap.add_argument("--only", nargs="+", choices=sorted(FIGURES), default=sorted(FIGURES))
    ap.add_argument("--out-dir", default=os.path.join("paper", "figures"))
    a = ap.parse_args(argv)

    cfg = load_config()
    figstyle.apply()
    out_dir = abs_path(a.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    made, skipped = [], []
    for name in a.only:
        path = FIGURES[name](cfg, os.path.join(out_dir, name))
        if path:
            made.append(path)
            print(f"  + {os.path.basename(path)}")
        else:
            skipped.append(name)
            print(f"  - {name}: inputs missing")
    print(f"\n{len(made)} figure(s) in {out_dir}" + (f", {len(skipped)} skipped" if skipped else ""))
    return 0 if made else 1
