"""Figure style: matplotlib's built-in tableau-colorblind10, default look otherwise."""
import os
from typing import Dict, Iterable, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WIDTH_SINGLE = 3.35  # inches, one journal column
WIDTH_DOUBLE = 6.90

# tableau-colorblind10, named
COLOURS = {
    "blue": "#006BA4", "orange": "#FF800E", "grey": "#595959", "lightgrey": "#ABABAB",
    "lightblue": "#5F9ED1", "darkorange": "#C85200", "paleblue": "#A2C8EC", "paleorange": "#FFBC79",
    "midgrey": "#898989",
}
ARM_COLOUR: Dict[str, str] = {
    "morpho_mix": COLOURS["orange"],
    "basic": COLOURS["grey"],
    "hed_jitter": COLOURS["blue"],
    "randstainna": COLOURS["lightblue"],
    "stain_mixup": COLOURS["lightgrey"],
    "dinobloom_s": "#7B3294",  # purple, outside the augmentation palette: the probes are not augmentations
    "dinobloom_b": "#C2A5CF",
}
ARM_LABEL: Dict[str, str] = {
    "morpho_mix": "MorphoMix", "basic": "Basic", "hed_jitter": "HED jitter",
    "randstainna": "RandStainNA", "stain_mixup": "Stain Mix-up", "dinobloom_s": "DinoBloom-S probe",
    "dinobloom_b": "DinoBloom-B probe",
}


def apply() -> None:
    """Matplotlib defaults with the tableau-colorblind10 cycle and print-ready output."""
    plt.style.use(["default", "tableau-colorblind10"])
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,  # TrueType, not Type 3
        "ps.fonttype": 42,
    })


def finish(fig, path_no_ext: str, formats: Iterable[str] = ("png", "pdf")) -> Tuple[str, ...]:
    """Save one figure in several formats and close it."""
    os.makedirs(os.path.dirname(path_no_ext) or ".", exist_ok=True)
    written = []
    for ext in formats:
        out = f"{path_no_ext}.{ext}"
        fig.savefig(out)
        written.append(out)
    plt.close(fig)
    return tuple(written)
