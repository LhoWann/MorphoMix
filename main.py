"""
MorphoMix entry point (run from the repo root): `python main.py <command> [--help]`.

Data
    prepare            cohorts, MLL23 style bank, RandStainNA fit                   scripts/data.py
    split              Aria / ALL-IDB2 groups for the bootstrap                     scripts/data.py
Training
    pretrain           train one arm and score the test cohorts                     scripts/train.py
    ablation           component ablation and control arms                         scripts/experiments.py
    screen             selection-cohort-only screening                              scripts/experiments.py
    tune               exploratory tuning with test scores (disclosed)              scripts/experiments.py
    foundation         frozen DinoBloom probes                                      scripts/foundation.py
    all                the whole protocol, resumable                                scripts/run_all.py
Evaluation
    tables             result tables from the stored artifacts                      scripts/evaluate.py
    rescore            re-score checkpoints with another inference setting          scripts/evaluate.py
    stats              per-seed paired bootstrap tests                              scripts/evaluate.py
    analysis           extended metrics and method-level tests                      scripts/evaluate.py
    stress             stress suite on the selection cohort                         scripts/evaluate.py
    profile            parameters, GFLOPs and latency                               scripts/evaluate.py
    xai                Layer-CAM localisation on ALL-IDB2                           scripts/evaluate.py
    paperstats         paper statistics from the stored predictions and logs        scripts/paper_stats.py
Audits
    shortcut           background reference, residual AUC, background swap          scripts/audit.py
    leakaudit          exact and perceptual duplicates between cohorts              scripts/audit.py
    fieldaudit         Aria detector and field-scoring audit                        scripts/audit.py
    confirm            pre-specified replicate (LeukemiaAttri L_100X_C1)            scripts/audit.py
Figures and utilities
    figures            augmentation panels                                          scripts/figures.py
    paperfigs          paper figures from the result tables                         scripts/figures.py
    resources          CPU / RAM / GPU use and bottlenecks per run                  src/utils/resources.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

COMMANDS = {
    "prepare": ("scripts.data", "prepare_binary_datasets", False),
    "split": ("scripts.data", "split_main", True),
    "pretrain": ("scripts.train", "main", True),
    "ablation": ("scripts.experiments", "ablation_main", True),
    "screen": ("scripts.experiments", "screen_main", True),
    "tune": ("scripts.experiments", "tune_main", True),
    "tables": ("scripts.evaluate", "tables_main", True),
    "rescore": ("scripts.evaluate", "rescore_main", True),
    "stats": ("scripts.evaluate", "stats_main", True),
    "stress": ("scripts.evaluate", "stress_main", True),
    "profile": ("scripts.evaluate", "profile_main", True),
    "xai": ("scripts.evaluate", "xai_main", True),
    "analysis": ("scripts.evaluate", "analysis_main", True),
    "foundation": ("scripts.foundation", "foundation_main", True),
    "shortcut": ("scripts.audit", "shortcut_main", True),
    "paperstats": ("scripts.paper_stats", "paper_stats_main", True),
    "fieldaudit": ("scripts.audit", "field_audit_main", True),
    "confirm": ("scripts.audit", "confirm_main", True),
    "leakaudit": ("scripts.audit", "leak_audit_main", True),
    "figures": ("scripts.figures", "generate_augmentation_figures", False),
    "paperfigs": ("scripts.figures", "paperfigs_main", True),
    "all": ("scripts.run_all", "main", True),
    "resources": ("src.utils.resources", "main", True),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv[0].lstrip("-") if argv else ""      # `-all` / `--all` work like `all`
    if not argv or argv[0] in ("-h", "--help") or command not in COMMANDS:
        print(__doc__)
        return 0 if argv and argv[0] in ("-h", "--help") else 1
    module_name, func_name, takes_argv = COMMANDS[command]
    if not takes_argv and argv[1:]:  # these commands take no options; `prepare --help` must not rebuild data/
        print(__doc__)
        return 0 if argv[1] in ("-h", "--help") else 1
    import importlib
    fn = getattr(importlib.import_module(module_name), func_name)
    return fn(argv[1:]) if takes_argv else fn()


if __name__ == "__main__":
    sys.exit(main() or 0)
