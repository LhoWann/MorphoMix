"""Resource monitor: CPU, RAM, /dev/shm and GPU samples from a background thread, per-epoch summaries with the
data-loader wait, and `python main.py resources`, which flags the bottleneck of each stage.

The sampler only reads counters (psutil, NVML or `nvidia-smi`, the torch allocator statistics): it touches no RNG
stream and never synchronises a CUDA stream, so a fixed-seed run stays bit-identical with the monitor on.
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import threading
import time
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Optional

import torch

from src.utils.config import abs_path, load_config
from src.utils.logger import boxed_table, get_console

try:
    import psutil
except ImportError:
    psutil = None
try:
    import pynvml
except ImportError:
    pynvml = None

SUMMARY_FILE, SAMPLES_FILE = "resources.csv", "resources_samples.csv"  # under {results_dir}/logs
GB = 2 ** 30
NAN = float("nan")
SATURATED_CORE = 90.0  # % busy for a core to count as saturated
# bottleneck flags of `python main.py resources`
LOW_GPU_UTIL, HIGH_DATA_WAIT, LOW_VRAM, LOW_RAM_AVAIL, HIGH_SHM, HIGH_CPU = 70.0, 0.20, 0.50, 0.10, 0.80, 90.0

SUMMARY_HEADER = [
    "timestamp", "run_id", "stage", "epoch", "epoch_s",
    "train_s", "train_steps", "train_step_ms", "train_wait_ms", "train_wait_frac", "val_s", "val_wait_frac",
    "gpu_source", "gpu_util_train", "gpu_util_val", "gpu_mem_peak_gb", "gpu_mem_total_gb",
    "torch_alloc_peak_gb", "torch_reserved_peak_gb",
    "cpu_count", "proc_cpu_cores_mean", "proc_cpu_cores_max", "sys_cpu_mean", "sys_cpu_max", "cores_saturated_mean",
    "proc_rss_peak_gb", "ram_used_peak_gb", "ram_avail_min_gb", "ram_total_gb", "shm_used_peak_gb", "shm_total_gb",
    "samples", "sampler_ms",
]
SAMPLE_HEADER = [
    "timestamp", "run_id", "epoch", "phase", "proc_cpu_cores", "sys_cpu", "cores_saturated", "core_max",
    "proc_rss_gb", "ram_used_gb", "ram_avail_gb", "shm_used_gb", "gpu_util", "gpu_mem_gb", "torch_alloc_gb",
    "torch_reserved_gb", "sampler_ms",
]


def _finite(values: Iterable[float]) -> List[float]:
    return [v for v in values if v == v]


def _mean(values: Iterable[float]) -> float:
    values = _finite(values)
    return sum(values) / len(values) if values else NAN


def _append_csv(path: str, header: List[str], rows: List[Dict]) -> None:
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if new:
            writer.writeheader()
        writer.writerows({k: f"{v:.4g}" if isinstance(v, float) else v for k, v in row.items()} for row in rows)


class ResourceMonitor:
    """Samples every `interval` s while active (`with monitor:`); `timed` wraps a loader to measure the data wait.

    Epoch protocol: `begin_epoch(n)`, iterate loaders through `timed(loader, "train" | "val")`, then `end_epoch()`
    appends one row to resources.csv (and the raw samples to resources_samples.csv) and returns a short console
    suffix. A missing source (psutil, NVML, nvidia-smi, CUDA) leaves its columns NaN. `config` is the `resources:`
    block; None or `enabled: false` turns every method into a no-op and `timed` into the identity.
    """

    def __init__(self, log_dir: str = "", run_id: str = "", stage: str = "", config: Optional[Dict] = None):
        config = config or {}
        self.enabled = bool(config.get("enabled", False))
        self.interval = float(config.get("interval", 5.0))
        self.raw_samples = bool(config.get("raw_samples", True))
        self.log_dir, self.run_id, self.stage = log_dir, run_id, stage
        self.epoch = 0
        self.phase = "idle"
        self.wait: Dict[str, float] = defaultdict(float)  # s spent in next(loader), per phase
        self.busy: Dict[str, float] = defaultdict(float)  # wall s of the loader loop, per phase
        self.steps: Dict[str, int] = defaultdict(int)
        self.rows: List[Dict] = []
        self._epoch_start = 0.0
        self._samples: List[Dict] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc = psutil.Process() if psutil is not None else None
        self._procs: Dict[int, "psutil.Process"] = {}  # cached so cpu_percent() measures since the last sample
        self._cuda = False
        self._device = -1
        self._uuid = ""
        self._nvml = None
        self.gpu_source = "none"

    def __enter__(self) -> "ResourceMonitor":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        self._cuda = torch.cuda.is_available()
        if self._cuda:
            self._device = torch.cuda.current_device()
            self._uuid = f"GPU-{torch.cuda.get_device_properties(self._device).uuid}"
        self.gpu_source = self._open_gpu()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="resource-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None
        if self._nvml is not None:
            pynvml.nvmlShutdown()
            self._nvml = None

    def _open_gpu(self) -> str:
        if not self._cuda:
            return "none"
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                try:
                    self._nvml = pynvml.nvmlDeviceGetHandleByUUID(self._uuid)
                    return "nvml"
                except pynvml.NVMLError:
                    pynvml.nvmlShutdown()
                    raise
            except pynvml.NVMLError:
                self._nvml = None
        return "nvidia-smi" if shutil.which("nvidia-smi") else "torch"

    def _loop(self) -> None:
        self._sample()  # discarded: psutil's first cpu_percent() calls only set the reference point
        while not self._stop.wait(self.interval):
            sample = self._sample()
            with self._lock:
                self._samples.append(sample)

    def _proc_tree(self) -> List["psutil.Process"]:
        """The main process and its children (DataLoader workers), with Process objects reused across samples."""
        try:
            live = [self._proc, *self._proc.children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            live = [self._proc]
        self._procs = {p.pid: self._procs.get(p.pid, p) for p in live}
        return list(self._procs.values())

    def _gpu(self) -> tuple:
        """(utilisation %, memory used GB) of the device, NaN when the source cannot tell."""
        if self.gpu_source == "nvml":
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml).gpu
                return float(util), pynvml.nvmlDeviceGetMemoryInfo(self._nvml).used / GB
            except pynvml.NVMLError:
                return NAN, NAN
        if self.gpu_source == "nvidia-smi":
            query = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits",
                     "-i", self._uuid]
            try:
                out = subprocess.run(query, capture_output=True, text=True, timeout=10, check=True).stdout
                util, used = (float(v) for v in out.strip().split(","))
                return util, used / 1024
            except (subprocess.SubprocessError, OSError, ValueError):
                return NAN, NAN
        if self.gpu_source == "torch":
            free, total = torch.cuda.mem_get_info(self._device)
            return NAN, (total - free) / GB
        return NAN, NAN

    def _sample(self) -> Dict:
        t0 = time.perf_counter()
        row = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "run_id": self.run_id, "epoch": self.epoch,
               "phase": self.phase}
        row.update({k: NAN for k in SAMPLE_HEADER if k not in row})
        if psutil is not None:
            cpu = rss = 0.0
            for p in self._proc_tree():
                try:
                    cpu += p.cpu_percent()
                    rss += p.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            cores = psutil.cpu_percent(percpu=True)
            vm = psutil.virtual_memory()
            row.update(proc_cpu_cores=cpu / 100, sys_cpu=sum(cores) / len(cores), core_max=max(cores),
                       cores_saturated=sum(c >= SATURATED_CORE for c in cores), proc_rss_gb=rss / GB,
                       ram_used_gb=(vm.total - vm.available) / GB, ram_avail_gb=vm.available / GB)
        if os.path.isdir("/dev/shm"):
            row["shm_used_gb"] = shutil.disk_usage("/dev/shm").used / GB
        row["gpu_util"], row["gpu_mem_gb"] = self._gpu()
        if self._cuda:  # allocator counters only: no stream synchronisation
            row["torch_alloc_gb"] = torch.cuda.memory_allocated(self._device) / GB
            row["torch_reserved_gb"] = torch.cuda.memory_reserved(self._device) / GB
        row["sampler_ms"] = (time.perf_counter() - t0) * 1000
        return row

    def timed(self, iterable: Iterable, phase: str) -> Iterable:
        """`iterable` with the time spent waiting for each item added to `wait[phase]`."""
        return self._timed(iterable, phase) if self.enabled else iterable

    def _timed(self, iterable: Iterable, phase: str) -> Iterator:
        self.phase = phase
        start = time.perf_counter()
        iterator = iter(iterable)  # starts the workers of a non-persistent loader: counted as wait
        self.wait[phase] += time.perf_counter() - start
        try:
            while True:
                t0 = time.perf_counter()
                try:
                    item = next(iterator)
                except StopIteration:
                    return
                finally:
                    self.wait[phase] += time.perf_counter() - t0
                self.steps[phase] += 1
                yield item
        finally:
            self.busy[phase] += time.perf_counter() - start
            self.phase = "idle"

    def begin_epoch(self, epoch: int) -> None:
        if not self.enabled:
            return
        self.epoch = epoch
        self.wait.clear()
        self.busy.clear()
        self.steps.clear()
        if self._cuda:
            torch.cuda.reset_peak_memory_stats(self._device)
        self._epoch_start = time.perf_counter()

    def end_epoch(self) -> str:
        """Writes the epoch row and returns its console suffix (`| gpu 87% | vram ... | data-wait 4%`)."""
        if not self.enabled:
            return ""
        with self._lock:
            samples, self._samples = self._samples, []

        def col(key: str, phase: Optional[str] = None) -> List[float]:
            return _finite(s[key] for s in samples if phase is None or s["phase"] == phase)

        def busy_frac(phase: str) -> float:
            return self.wait[phase] / self.busy[phase] if self.busy[phase] else NAN

        steps, cuda, dev = self.steps["train"], self._cuda, self._device
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "run_id": self.run_id, "stage": self.stage,
            "epoch": self.epoch, "epoch_s": time.perf_counter() - self._epoch_start,
            "train_s": self.busy["train"], "train_steps": steps,
            "train_step_ms": 1000 * self.busy["train"] / steps if steps else NAN,
            "train_wait_ms": 1000 * self.wait["train"] / steps if steps else NAN,
            "train_wait_frac": busy_frac("train"), "val_s": self.busy["val"], "val_wait_frac": busy_frac("val"),
            "gpu_source": self.gpu_source,
            "gpu_util_train": _mean(col("gpu_util", "train")), "gpu_util_val": _mean(col("gpu_util", "val")),
            "gpu_mem_peak_gb": max(col("gpu_mem_gb"), default=NAN),
            "gpu_mem_total_gb": torch.cuda.get_device_properties(dev).total_memory / GB if cuda else NAN,
            "torch_alloc_peak_gb": torch.cuda.max_memory_allocated(dev) / GB if cuda else NAN,
            "torch_reserved_peak_gb": torch.cuda.max_memory_reserved(dev) / GB if cuda else NAN,
            "cpu_count": os.cpu_count(),
            "proc_cpu_cores_mean": _mean(col("proc_cpu_cores")),
            "proc_cpu_cores_max": max(col("proc_cpu_cores"), default=NAN),
            "sys_cpu_mean": _mean(col("sys_cpu")), "sys_cpu_max": max(col("sys_cpu"), default=NAN),
            "cores_saturated_mean": _mean(col("cores_saturated")),
            "proc_rss_peak_gb": max(col("proc_rss_gb"), default=NAN),
            "ram_used_peak_gb": max(col("ram_used_gb"), default=NAN),
            "ram_avail_min_gb": min(col("ram_avail_gb"), default=NAN),
            "ram_total_gb": psutil.virtual_memory().total / GB if psutil is not None else NAN,
            "shm_used_peak_gb": max(col("shm_used_gb"), default=NAN),
            "shm_total_gb": shutil.disk_usage("/dev/shm").total / GB if os.path.isdir("/dev/shm") else NAN,
            "samples": len(samples), "sampler_ms": _mean(col("sampler_ms")),
        }
        self.rows.append(row)
        _append_csv(os.path.join(self.log_dir, SUMMARY_FILE), SUMMARY_HEADER, [row])
        if self.raw_samples and samples:
            _append_csv(os.path.join(self.log_dir, SAMPLES_FILE), SAMPLE_HEADER, samples)
        return (f" | gpu {row['gpu_util_train']:.0f}% | vram {row['torch_reserved_peak_gb']:.1f}/"
                f"{row['gpu_mem_total_gb']:.0f} GB | data-wait {row['train_wait_frac']:.0%}")

    def run_line(self) -> str:
        """One console line for the whole run, from `aggregate` over its epochs."""
        if not self.rows:
            return ""
        r = aggregate(self.rows)
        val = f" val {r['val_wait_frac']:.0%}" if r["val_wait_frac"] == r["val_wait_frac"] else ""
        return (f"{self.run_id} resources: gpu {r['gpu_util_train']:.0f}% | vram peak {r['vram_peak_gb']:.1f}/"
                f"{r['gpu_mem_total_gb']:.0f} GB | data-wait train {r['train_wait_frac']:.0%}{val}"
                f" | cpu {r['proc_cpu_cores_mean']:.1f} cores, system "
                f"{r['sys_cpu_mean']:.0f}% | ram peak {r['ram_used_peak_gb']:.1f}/{r['ram_total_gb']:.0f} GB")


MEAN_KEYS = ("epoch_s", "train_step_ms", "train_wait_ms", "train_wait_frac", "val_wait_frac", "gpu_util_train",
             "gpu_util_val", "proc_cpu_cores_mean", "sys_cpu_mean", "cores_saturated_mean", "sampler_ms", "cpu_count")
PEAK_KEYS = ("gpu_mem_peak_gb", "torch_alloc_peak_gb", "torch_reserved_peak_gb", "proc_rss_peak_gb",
             "ram_used_peak_gb", "shm_used_peak_gb", "gpu_mem_total_gb", "ram_total_gb", "shm_total_gb")


def aggregate(rows: List[Dict]) -> Dict:
    """Epoch rows -> means over the epochs after the first (it includes the worker start-up), peaks over all,
    bottleneck flags."""
    steady = [r for r in rows if r["epoch"] > 1] or rows
    out = {"stage": rows[0]["stage"], "runs": len({r["run_id"] for r in rows}), "epochs": len(rows)}
    out.update({k: _mean(r[k] for r in steady) for k in MEAN_KEYS})
    out.update({k: max(_finite(r[k] for r in rows), default=NAN) for k in PEAK_KEYS})
    out["ram_avail_min_gb"] = min(_finite(r["ram_avail_min_gb"] for r in rows), default=NAN)
    out["vram_peak_gb"] = max(_finite((out["torch_reserved_peak_gb"], out["gpu_mem_peak_gb"])), default=NAN)
    out["flags"] = bottlenecks(out)
    return out


def bottlenecks(row: Dict) -> List[str]:
    """Flags of one stage summary (means over epochs, peaks as maxima)."""
    flags = []
    if row["gpu_util_train"] < LOW_GPU_UTIL and row["train_wait_frac"] > HIGH_DATA_WAIT:
        flags.append("input-bound (num_workers, cheaper loading)")
    if row["val_wait_frac"] > HIGH_DATA_WAIT:
        flags.append("val input-bound (eval_num_workers)")
    if row["vram_peak_gb"] < LOW_VRAM * row["gpu_mem_total_gb"]:
        flags.append("vram < 50% (headroom)")
    if row["ram_avail_min_gb"] < LOW_RAM_AVAIL * row["ram_total_gb"]:
        flags.append("ram near limit")
    if row["shm_used_peak_gb"] > HIGH_SHM * row["shm_total_gb"]:
        flags.append("/dev/shm near limit")
    if row["sys_cpu_mean"] > HIGH_CPU:
        flags.append("cpu saturated")
    return flags


def summarise(log_dir: str) -> List[Dict]:
    """One row per run group (experiment id without the seed): epoch means, memory peaks, bottleneck flags."""
    path = os.path.join(log_dir, SUMMARY_FILE)
    if not os.path.exists(path):
        return []
    groups: Dict[str, List[Dict]] = defaultdict(list)
    text = ("timestamp", "run_id", "stage", "gpu_source")
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if set(r) != set(SUMMARY_HEADER):  # an older schema or a torn line
                continue
            try:
                row = {k: (v if k in text else float(v)) for k, v in r.items()}
            except (TypeError, ValueError):
                continue
            groups[re.sub(r"_seed\d+$", "", row["run_id"])].append(row)
    return [{"group": name, **aggregate(rows)} for name, rows in groups.items()]


def print_summary(log_dir: str, console) -> List[Dict]:
    """Rich table of `summarise`; nothing when no run has logged resources yet."""
    rows = summarise(log_dir)
    if not rows:
        return rows
    table = boxed_table("Resources per run group (means of epochs 2+, memory peaks)")
    for col in ("group", "epochs", "s/epoch", "step ms", "wait tr", "wait val", "gpu tr", "gpu val", "vram GB",
                "cpu cores", "sys cpu", "ram GB", "shm GB"):
        table.add_column(col, justify="left" if col == "group" else "right")

    def fmt(value: float, spec: str) -> str:
        return "-" if value != value else format(value, spec)

    for r in rows:
        table.add_row(
            r["group"], str(r["epochs"]), fmt(r["epoch_s"], ".0f"), fmt(r["train_step_ms"], ".0f"),
            fmt(r["train_wait_frac"], ".0%"), fmt(r["val_wait_frac"], ".0%"), fmt(r["gpu_util_train"], ".0f"),
            fmt(r["gpu_util_val"], ".0f"), f"{fmt(r['vram_peak_gb'], '.1f')}/{fmt(r['gpu_mem_total_gb'], '.0f')}",
            f"{fmt(r['proc_cpu_cores_mean'], '.1f')}/{fmt(r['cpu_count'], '.0f')}", fmt(r["sys_cpu_mean"], ".0f"),
            f"{fmt(r['ram_used_peak_gb'], '.1f')}/{fmt(r['ram_total_gb'], '.0f')}",
            f"{fmt(r['shm_used_peak_gb'], '.1f')}/{fmt(r['shm_total_gb'], '.0f')}",
        )
    console.print(table)
    for r in rows:
        console.print(f"  {r['group']}: {'; '.join(r['flags']) or 'no bottleneck flagged'}")
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Per-run-group resource summary and bottleneck flags from "
                                            f"{{results_dir}}/logs/{SUMMARY_FILE}.")
    p.add_argument("--results-dir", default=None, help="default: results_dir of the config")
    a = p.parse_args(argv)
    log_dir = os.path.join(abs_path(a.results_dir or load_config()["results_dir"]), "logs")
    if not print_summary(log_dir, get_console()):
        print(f"no {SUMMARY_FILE} under {log_dir}")
        return 1
    print(f"flags: input-bound = gpu < {LOW_GPU_UTIL:.0f}% with data-wait > {HIGH_DATA_WAIT:.0%}; vram < "
          f"{LOW_VRAM:.0%} of the card; ram available < {LOW_RAM_AVAIL:.0%}; /dev/shm > {HIGH_SHM:.0%}; "
          f"system cpu > {HIGH_CPU:.0f}%")
    return 0
