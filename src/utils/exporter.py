"""Publication table exporter."""
import json
import os
import re
from typing import Dict, List, Optional


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.1f}" if abs(v) >= 10 else f"{v:.3f}"
    return str(v)


TEX_ESCAPES = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", '"': r"\textquotedbl{}", "±": r"$\pm$", "+/-": r"$\pm$",
    ">=": r"$\geq$", "<=": r"$\leq$", "<": r"$<$", ">": r"$>$",
}
TEX_SPECIAL = re.compile("|".join(re.escape(k) for k in sorted(TEX_ESCAPES, key=len, reverse=True)))


def _tex_escape(s: str) -> str:
    """One pass, so a replacement is never escaped again."""
    return TEX_SPECIAL.sub(lambda m: TEX_ESCAPES[m.group()], s)


def export_results_table(
    rows: List[Dict],
    output_prefix: str,
    caption: str = "",
    label: str = "",
    columns: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Writes `.md` and `.tex` with `columns` (default: all) and `.json` with every field."""
    if not rows:
        raise ValueError("export_results_table: no rows")
    cols = columns or list(rows[0].keys())
    os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)

    md_lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        md_lines.append("| " + " | ".join(_fmt(r.get(c, "")) for c in cols) + " |")
    md = "\n".join(md_lines) + "\n"
    if caption:
        md = f"**{caption}**\n\n" + md

    align = "".join("r" if all(isinstance(r.get(c), (int, float)) for r in rows) else "l" for c in cols)
    tex = [
        r"\begin{table}[htbp]", r"\centering",
        (r"\caption{" + _tex_escape(caption) + "}") if caption else "",
        (r"\label{" + label + "}") if label else "",
        r"\begin{tabular}{" + align + "}", r"\toprule",
        " & ".join(_tex_escape(c) for c in cols) + r" \\", r"\midrule",
    ]
    for r in rows:
        tex.append(" & ".join(_tex_escape(_fmt(r.get(c, ""))) for c in cols) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    paths = {
        "md": output_prefix + ".md",
        "tex": output_prefix + ".tex",
        "json": output_prefix + ".json",
    }
    with open(paths["md"], "w", encoding="utf-8") as f:
        f.write(md)
    with open(paths["tex"], "w", encoding="utf-8") as f:
        f.write("\n".join(line for line in tex if line is not None))
    with open(paths["json"], "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    return paths
