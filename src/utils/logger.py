"""Rich console output in the style of PyTorch Lightning's RichProgressBar; only the bar colour differs."""
import sys
import os
import stat
import logging
import warnings
from datetime import timedelta
from typing import Dict, Any, Optional

# Suppress library warnings and noisy logs
warnings.filterwarnings("ignore")
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
os.environ["ACCELERATE_LOG_LEVEL"] = "ERROR"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
try:
    import huggingface_hub.utils.logging as hlog
    hlog.set_verbosity_error()
except Exception:
    pass

# Force UTF-8 and Code Page 65001 on Windows
if sys.platform == "win32":
    try:
        os.system("chcp 65001 >nul 2>&1")
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console
from rich.theme import Theme
from rich.table import Table
from rich.progress import BarColumn, Progress, ProgressColumn, Task, TextColumn
from rich.text import Text
from rich import box

BAR_COLOR = "#E91E8C"  # pink magenta: progress bar, table titles and headers


def _stdout_is_file() -> bool:
    try:
        return stat.S_ISREG(os.fstat(sys.stdout.fileno()).st_mode)
    except (OSError, ValueError, AttributeError):
        return False


# Colours are forced on for Colab's `!python` pipe (it renders ANSI colours) but not for a log file. Live bars need
# cursor movement, which only a real terminal honours; elsewhere Colab stacks every redraw as a new line, so the
# Trainer prints one line per epoch instead.
_console = Console(theme=Theme({
    "accent": "default",
    "info": "default",
    "success": "default",
    "highlight": "bold",
    "warning": "yellow",
    "error": "red",
}), force_terminal=not _stdout_is_file(), force_interactive=sys.stdout.isatty(),
    width=None if sys.stdout.isatty() else 140, highlight=False)


def get_console() -> Console:
    return _console


def boxed_table(title: str = "", show_header: bool = True, width: Optional[int] = None) -> Table:
    """The one table style: plain rounded border, bold magenta header and title."""
    return Table(title=f"[bold]{title}[/bold]" if title else None, title_justify="left", title_style=BAR_COLOR,
                 box=box.ROUNDED, header_style=f"bold {BAR_COLOR}",
                 show_header=show_header, padding=(0, 1), width=width)


def print_header_panel(title: str, subtitle: str, info_dict: Optional[Dict[str, Any]] = None) -> None:
    """Run header as a two-column table."""
    table = boxed_table(title, show_header=False)
    table.add_column(style="dim", no_wrap=True)
    table.add_column()
    for key, value in (info_dict or {}).items():
        table.add_row(str(key), str(value))
    if subtitle:
        table.caption, table.caption_justify, table.caption_style = subtitle, "left", "dim"
    _console.print()
    _console.print(table)


def print_ablation_header(step_idx: int, total_steps: int, name: str, desc: str) -> None:
    """One line per run: `> name  3/9 - description`."""
    counter = f"{step_idx}/{total_steps} - " if total_steps else ""
    _console.print(f"\n[accent]>[/accent] [bold]{name}[/bold]  [dim]{counter}{desc}[/dim]")


# Columns and theme of Lightning's RichProgressBar (lightning/pytorch/callbacks/progress/rich_progress.py),
# with the bar colour swapped for BAR_COLOR.
VALIDATION_DESCRIPTION = "Validation"


class _BatchesProcessedColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        return Text(f"{int(task.completed)}/{task.total}")


class _TimeColumn(ProgressColumn):
    max_refresh = 0.5  # refresh twice a second to prevent jitter

    def render(self, task: Task) -> Text:
        elapsed = task.finished_time if task.finished else task.elapsed
        remaining = task.time_remaining
        elapsed_delta = "-:--:--" if elapsed is None else str(timedelta(seconds=int(elapsed)))
        remaining_delta = "-:--:--" if remaining is None else str(timedelta(seconds=int(remaining)))
        return Text(f"{elapsed_delta} • {remaining_delta}", style="dim")


class _SpeedColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        speed = f"{task.speed:>.2f}" if task.speed is not None else "0.00"
        return Text(f"{speed}it/s", style="dim underline")


def create_progress_bar(console: Optional[Console] = None) -> Progress:
    """One live display per run: the reused `Epoch x/N` bar plus a transient `Validation` bar."""
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style=BAR_COLOR, finished_style=BAR_COLOR, pulse_style=BAR_COLOR),
        _BatchesProcessedColumn(),
        _TimeColumn(),
        _SpeedColumn(),
        TextColumn("{task.fields[metrics]}", style="italic"),
        console=console or _console,
    )


def epoch_description(epoch: int, epochs: int) -> str:
    """`Epoch x/N` for the zero-based loop index `epoch` (1/30 ... 30/30, as the per-epoch log lines), padded to the
    width of the validation bar's label."""
    return f"{f'Epoch {epoch + 1}/{epochs}':{len(VALIDATION_DESCRIPTION)}}"


def format_metrics(metrics: Dict[str, float]) -> str:
    return " ".join(f"{name}: {value:.3f}" for name, value in metrics.items())


def print_master_comparison(results_list: list) -> None:
    """Run summary across arms."""
    table = boxed_table("Summary")
    for col, justify in (("arm", "left"), ("best epoch", "right"), ("val macro-F1", "right"),
                         ("accuracy", "right"), ("balanced acc", "right"), ("val loss", "right")):
        table.add_column(col, justify=justify)
    for r in results_list:
        name = r.get("augmentation", "")
        style = "bold" if name.startswith("morpho") else None
        table.add_row(
            name,
            str(r.get("best_epoch", 0)),
            f"{r.get('macro_f1', 0):.4f}",
            f"{r.get('accuracy', 0):.4f}",
            f"{r.get('balanced_accuracy', 0):.4f}",
            f"{r.get('best_val_loss', 0):.4f}",
            style=style,
        )
    _console.print(table)
