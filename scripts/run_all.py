"""End-to-end runner for the whole MorphoMix protocol."""
import argparse
import copy
import glob
import json
import os
import time
import traceback
from typing import Callable, Dict, List, Optional

from src.utils.config import abs_path, apply_overrides, load_config, refuse_real_results_dir, setup_cuda_env
setup_cuda_env()

from rich.markup import escape  # noqa: E402
from rich.table import Table  # noqa: E402

from src.utils.logger import (  # noqa: E402
    BAR_COLOR, boxed_table, get_console, print_ablation_header, print_header_panel,
)
from scripts import audit, data, evaluate, experiments, figures, foundation, train  # noqa: E402
from scripts.train import seeds_for  # noqa: E402

# Screening (`main.py screen`) is not a stage: its verdicts are adopted into config.yaml by hand before this plan runs
STAGES = ["prepare", "phase1", "foundation", "tables", "ablation", "xai", "stats", "analysis", "shortcut", "stress",
          "figures", "profile"]
DEFAULT_STAGES = STAGES[1:]  # `prepare` needs data/raw, which is never shipped to Colab
# derived from experiments.ARMS so the two lists cannot drift
ABLATION_ARMS = sorted(experiments.ARMS)
DURATION_CSV = "logs/run_durations.csv"  # under results_dir
STATS_REFERENCE = "morpho_mix"  # `stats --reference` default, the arm every baseline is tested against
DURATION_HEADER = "timestamp,stage,cell,command,seconds,exit_code,status"
FINGERPRINT = "env_fingerprint.json"  # under results_dir, written by the first `all` run

STAGE_DESC = {
    "prepare": "data layout",
    "phase1": "binary training, tests scored once",
    "tables": "result tables",
    "ablation": "component ablation",
    "xai": "CAM localisation (ALL-IDB2)",
    "stats": "bootstrap tests",
    "foundation": "frozen DinoBloom + linear probe",
    "analysis": "extended and seed-level metrics",
    "shortcut": "background reference, residual AUC, background swap",
    "stress": "val stress suite",
    "figures": "augmentation figures",
    "profile": "backbone profile",
}


def env_fingerprint() -> Dict:
    import platform
    import accelerate
    import timm
    import torch
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "timm": timm.__version__, "accelerate": accelerate.__version__,
            "gpu": props.name if props else "CPU",
            "gpu_memory_gb": round(props.total_memory / 2 ** 30, 1) if props else 0}


def check_fingerprint(cfg: Dict, console) -> None:
    """Fixed-seed runs are bit-reproducible only on one GPU model and kernel set, so a results folder is continued
    only with the GPU name and the torch / cuDNN major.minor it was started with."""
    path = abs_path(os.path.join(cfg["results_dir"], FINGERPRINT))
    now = env_fingerprint()
    console.print(f"[dim]{', '.join(f'{k} {v}' for k, v in now.items())}[/dim]")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(now, f, indent=2)
        os.replace(path + ".tmp", path)
        return
    with open(path, encoding="utf-8") as f:
        first = json.load(f)

    def key(fp: Dict) -> Dict:
        cudnn = fp["cudnn"] or 0  # 90100 = 9.1.0 from cuDNN 9 on, 8907 = 8.9.7 before
        return {"gpu": fp["gpu"], "torch": ".".join(fp["torch"].split(".")[:2]),
                "cudnn": list(divmod(cudnn // 100, 100 if cudnn >= 10000 else 10))}

    differ = {k: (v, key(now)[k]) for k, v in key(first).items() if v != key(now)[k]}
    if differ:
        raise SystemExit(f"{cfg['results_dir']} was started with another environment (was, now): {differ}; use the "
                         f"same GPU model and library versions, or a new results_dir ({path})")


class Runner:
    """Executes cells, keeps the per-stage bookkeeping and prints the summary."""

    def __init__(self, cfg: Dict, force: bool, dry_run: bool, keep_going: bool):
        self.cfg = cfg
        self.force = force
        self.dry_run = dry_run
        self.keep_going = keep_going
        self.console = get_console()
        self.records: List[Dict] = []
        self.started = time.time()
        self._csv = abs_path(os.path.join(cfg["results_dir"], DURATION_CSV))

    def _exists(self, rel_paths: List[str]) -> bool:
        return bool(rel_paths) and all(os.path.exists(abs_path(p)) for p in rel_paths)

    def ckpt(self, experiment_id: str) -> str:
        return f"{self.cfg['results_dir']}/checkpoints/{experiment_id}_best.pt"

    def pred(self, experiment_id: str, test_key: str) -> str:
        return f"{self.cfg['results_dir']}/predictions/{experiment_id}_test_{test_key}.json"

    def _log_duration(self, stage: str, cell: str, command: str, seconds: float, code: int, status: str) -> None:
        path = self._csv
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(DURATION_HEADER + "\n")
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f'{stamp},{stage},{cell},"{command}",{seconds:.1f},{code},{status}\n')

    def _eta(self, stage: str, remaining: int) -> str:
        done = [r["seconds"] for r in self.records if r["stage"] == stage and r["status"] == "ok"]
        if not done or remaining <= 0:
            return ""
        mean = sum(done) / len(done)
        return f", {remaining} left, ~{self._fmt(mean * remaining)}"

    @staticmethod
    def _fmt(seconds: float) -> str:
        seconds = int(round(seconds))
        return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"

    def cell(
        self,
        stage: str,
        name: str,
        command: str,
        fn: Callable[[], object],
        expect: Optional[List[str]] = None,
        desc: str = "",
        index: int = 0,
        total: int = 0,
        remaining: int = 0,
        done: Optional[bool] = None,
        check: Optional[Callable[[], None]] = None,
    ) -> None:
        """`done` (default: every `expect` file exists) skips the cell once `check`, the resume guard of a finished
        run, passes; a failed check, an exception, a SystemExit or a non-zero int returned by `fn` is a failed cell."""
        done = (not self.force) and (self._exists(expect or []) if done is None else done)

        if self.dry_run:
            self.console.print(f"  {name:<36}[dim]{'done, skipped' if done else command}[/dim]")
            return

        if not done:
            print_ablation_header(index, total, name, desc or STAGE_DESC.get(stage, stage))
        t0 = time.time()
        try:
            if done:
                if check is not None:
                    check()
                self.console.print(f"  {name:<36}[dim]done, skipped[/dim]")
                self._log_duration(stage, name, command, 0.0, 0, "skipped")
                self.records.append({"stage": stage, "cell": name, "seconds": 0.0, "status": "skipped"})
                return
            code = fn()
            if isinstance(code, int) and code != 0:
                raise RuntimeError(f"{command} returned exit code {code}")
        except KeyboardInterrupt:
            elapsed = time.time() - t0
            self._log_duration(stage, name, command, elapsed, 130, "interrupted")
            self.records.append({"stage": stage, "cell": name, "seconds": elapsed, "status": "interrupted"})
            raise
        except (Exception, SystemExit) as exc:  # noqa: BLE001 - reported, not swallowed; SystemExit: a stage's exit
            elapsed = time.time() - t0
            self._log_duration(stage, name, command, elapsed, 1, "failed")
            self.records.append({"stage": stage, "cell": name, "seconds": elapsed, "status": "failed"})
            self.console.print(f"  [error]failed[/error] {name}  {type(exc).__name__}: {escape(str(exc))}")
            self.console.print(f"[dim]{escape(traceback.format_exc(limit=3))}[/dim]")
            if not self.keep_going:
                raise
            return

        elapsed = time.time() - t0
        self._log_duration(stage, name, command, elapsed, 0, "ok")
        self.records.append({"stage": stage, "cell": name, "seconds": elapsed, "status": "ok"})
        self.console.print(
            f"  {name:<36}[dim]done in {self._fmt(elapsed)}{self._eta(stage, remaining)}[/dim]"
        )

    def rule(self, stage: str) -> None:
        self.console.print()
        self.console.rule(f"[bold {BAR_COLOR}]{stage}[/] [dim]{STAGE_DESC.get(stage, '')}[/dim]", style="dim",
                          align="left")

    def summary(self) -> int:
        table = boxed_table("Summary")
        table.add_column("stage")
        table.add_column("ran", justify="right")
        table.add_column("skipped", justify="right")
        table.add_column("failed", justify="right")
        table.add_column("time", justify="right")

        failed = 0
        for stage in STAGES:
            rows = [r for r in self.records if r["stage"] == stage]
            if not rows:
                continue
            ran = sum(1 for r in rows if r["status"] == "ok")
            skipped = sum(1 for r in rows if r["status"] == "skipped")
            bad = sum(1 for r in rows if r["status"] in ("failed", "interrupted"))
            failed += bad
            table.add_row(stage, str(ran), str(skipped), str(bad) if bad else "-",
                          self._fmt(sum(r["seconds"] for r in rows)))

        self.console.print()
        self.console.print(table)
        wall = self._fmt(time.time() - self.started)
        verdict = f"[error]{failed} failed[/error]" if failed else "done"
        self.console.print(f"{verdict}  [dim]{wall}, results in {self.cfg['results_dir']}/[/dim]")
        return 1 if failed else 0


def check_adopted(cfg: Dict) -> None:
    """Refuses a plan whose config lacks what `tune --grid final --adopt` chose (tuning/adopted.json): resume never
    compares the recipe, so the remaining MorphoMix cells would silently train the un-adopted composition."""
    path = abs_path(f"{cfg['results_dir']}/tuning/adopted.json")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        adopted = json.load(f)
    for key, value in adopted["overrides"].items():
        current = cfg
        for part in key.split("."):
            current = current[part]
        if current != value:
            raise SystemExit(f"{key} is {current} in the config, {value} in {path} ({adopted['chosen']}); run "
                             f"`main.py tune --grid final --adopt` first")


def stage_prepare(run: Runner) -> None:
    run.cell("prepare", "prepare", "main.py prepare", data.prepare_binary_datasets,
             expect=["data/processed/cohorts.json"], index=1, total=1)


def stage_phase1(run: Runner, augs: List[str], seeds: List[int], epochs: Optional[int], limit: int) -> None:
    cfg = run.cfg
    epochs = epochs or cfg["phase1"]["epochs"]
    cells = [(a, s) for a in augs for s in seeds_for(cfg, a, seeds)]
    for i, (aug, seed) in enumerate(cells, start=1):
        eid = f"phase1_{aug}_seed{seed}"
        run.cell(
            "phase1", eid, f"main.py pretrain --aug {aug} --seed {seed}",
            lambda a=aug, s=seed: train.run_one(cfg, a, s, epochs, run.console, limit),
            expect=[run.ckpt(eid)] + [run.pred(eid, k) for k in train.TEST_SETS],
            check=lambda a=aug, e=eid: train.check_references(cfg, a, e),
            desc=train.experiment_desc(aug),
            index=i, total=len(cells), remaining=len(cells) - i,
        )


def stage_tables(run: Runner, augs: List[str], seeds: List[int]) -> None:
    def rebuild():
        evaluate.rebuild_phase1(run.cfg, augs, seeds, run.console)
        evaluate.background_reference(run.cfg, run.console)
    run.cell("tables", "rebuild", "main.py tables", rebuild, index=1, total=1)


def stage_ablation(run: Runner, arms: List[str], seeds: List[int], epochs: Optional[int], limit: int) -> None:
    done = set()
    runs_json = abs_path(f"{run.cfg['results_dir']}/tables/component_ablation_runs.json")
    if os.path.exists(runs_json) and not run.force:
        with open(runs_json, encoding="utf-8") as f:
            done = {(r["arm"], r["seed"]) for r in json.load(f)}

    cells = [(arm, seed) for seed in seeds for arm in arms]
    for i, (arm, seed) in enumerate(cells, start=1):
        argv = ["--arms", arm, "--seed", str(seed)]
        if epochs:
            argv += ["--epochs", str(epochs)]
        if limit:
            argv += ["--limit", str(limit)]
        command = "main.py ablation " + " ".join(argv)
        spec = experiments.ARMS[arm]
        tag = spec["aug"] if arm == experiments.REFERENCE_ARM else f"abl{arm}"
        arm_cfg, eid = apply_overrides(copy.deepcopy(run.cfg), spec["cfg"]), f"phase1_{tag}_seed{seed}"
        if arm != experiments.REFERENCE_ARM and spec["aug"] == experiments.ARM_AUG and arm_cfg == run.cfg:
            run.console.print(f"  abl{arm}_seed{seed:<30}[dim]same config as arm a (adopted), not trained[/dim]")
            continue
        run.cell(
            "ablation", f"abl{arm}_seed{seed}", command,
            lambda a=argv: experiments.ablation_main(a),
            done=(arm, seed) in done,
            check=lambda c=arm_cfg, a=spec["aug"], e=eid: train.check_references(c, a, e),
            desc=spec["desc"], index=i, total=len(cells), remaining=len(cells) - i,
        )


def stage_xai(run: Runner, augs: List[str], seed: int, limit: int) -> None:
    missing = [a for a in augs if not os.path.exists(abs_path(run.ckpt(f"phase1_{a}_seed{seed}")))]
    argv = ["--seed", str(seed), "--arms"] + list(augs) + (["--limit", str(limit)] if limit else [])

    def xai():
        if missing:  # a failed cell, so --continue-on-error lists it in the summary
            raise FileNotFoundError(f"no seed-{seed} checkpoint for {', '.join(missing)}")
        evaluate.xai_main(argv)
    run.cell("xai", f"xai_seed{seed}", "main.py xai " + " ".join(argv), xai,
             expect=[f"{run.cfg['results_dir']}/tables/xai_localization_arms.json"], index=1, total=1)


def stage_stats(run: Runner, n_boot: int) -> None:
    table = f"{run.cfg['results_dir']}/tables/statistical_tests.json"
    expect = [table]
    if os.path.exists(abs_path(table)):  # redone when the compared predictions changed after the table was written
        with open(abs_path(table), encoding="utf-8") as f:
            tested = {(r["test_set"], r["seed"], r["baseline"]) for r in json.load(f)}
        files = evaluate.prediction_files(run.cfg)
        baselines = [x for x in run.cfg["augmentations"] if x != STATS_REFERENCE]
        stamp = os.path.getmtime(abs_path(table))
        newer = any(os.path.getmtime(p) > stamp for (_, aug, _), p in files.items() if aug in run.cfg["augmentations"])
        if newer or evaluate.stats_pairs(files, STATS_REFERENCE, baselines) != tested:
            expect = []
    argv = ["--n-boot", str(n_boot)]
    run.cell("stats", "stats", "main.py stats " + " ".join(argv),
             lambda: evaluate.stats_main(argv), expect=expect, index=1, total=1)


def stage_foundation(run: Runner, limit: int) -> None:
    eid = lambda name: f"phase1_{name}_seed{run.cfg['primary_seed']}"  # noqa: E731
    names = sorted(foundation.MODELS)
    for i, name in enumerate(names, start=1):
        argv = ["--models", name] + (["--limit", str(limit)] if limit else [])
        run.cell("foundation", eid(name), "main.py foundation " + " ".join(argv),
                 lambda a=argv: foundation.foundation_main(a),
                 expect=[run.pred(eid(name), k) for k in train.TEST_SETS], index=i, total=len(names),
                 remaining=len(names) - i)


def stage_analysis(run: Runner, n_boot: int) -> None:
    argv = ["--n-boot", str(n_boot)]
    run.cell("analysis", "analysis", "main.py analysis " + " ".join(argv), lambda: evaluate.analysis_main(argv),
             index=1, total=1)


def stage_shortcut(run: Runner) -> None:
    table = f"{run.cfg['results_dir']}/tables/shortcut_bg_swap_runs.csv"
    expect = [table]
    ckpts = glob.glob(abs_path(f"{run.cfg['results_dir']}/checkpoints/phase1_*_seed*_best.pt"))
    if os.path.exists(abs_path(table)) and any(os.path.getmtime(c) > os.path.getmtime(abs_path(table)) for c in ckpts):
        expect = []  # redone when a checkpoint is newer than the table
    run.cell("shortcut", "shortcut", "main.py shortcut", lambda: audit.shortcut_main([]), expect=expect,
             index=1, total=1)


def stage_stress(run: Runner, limit: int) -> None:
    table = f"{run.cfg['results_dir']}/tables/stress_selection_val.json"
    expect = [table]
    if os.path.exists(abs_path(table)):  # a table written while a cell had failed (--continue-on-error) is redone
        with open(abs_path(table), encoding="utf-8") as f:
            scored = {(r["arm"], r["seed"]) for r in json.load(f)}
        if any((arm, seed) not in scored for arm, seed, _ in evaluate.phase1_checkpoints(run.cfg)):
            expect = []
    argv = ["--limit", str(limit)] if limit else []
    run.cell("stress", "stress", " ".join(["main.py stress"] + argv), lambda: evaluate.stress_main(argv), expect=expect,
             index=1, total=1)


def stage_figures(run: Runner) -> None:
    run.cell("figures", "figures", "main.py figures", figures.generate_augmentation_figures,
             expect=[f"{run.cfg['results_dir']}/figures/augmentations/all_augmentations_comparison.png"],
             index=1, total=1)


def stage_profile(run: Runner) -> None:
    run.cell("profile", "profile", "main.py profile", lambda: evaluate.profile_main([]),
             expect=[f"{run.cfg['results_dir']}/tables/model_profile.json"], index=1, total=1)


def plan_table(cfg: Dict, stages: List[str], augs: List[str], seeds: List[int], arms: List[str],
               ablation_seeds: List[int]) -> Table:
    # runs = trainings: ablation arm (a) reuses the main morpho_mix run when that run is in the plan
    reused = experiments.REFERENCE_ARM in arms and "phase1" in stages and experiments.ARM_AUG in augs
    counts = {
        "phase1": sum(len(seeds_for(cfg, a, seeds)) for a in augs),
        "foundation": len(foundation.MODELS),
        "ablation": sum(1 for arm in arms for s in ablation_seeds
                        if not (reused and arm == experiments.REFERENCE_ARM
                                and s in seeds_for(cfg, experiments.ARM_AUG, seeds))),
    }
    chosen = [s for s in STAGES if s in stages]
    table = boxed_table("Plan")
    table.add_column("stage", footer="total")
    table.add_column("runs", justify="right", footer=str(sum(counts.get(s, 1) for s in chosen)))
    table.add_column("what", style="dim", ratio=1)
    table.show_footer = True
    table.footer_style = "bold"
    notes = {"ablation": f"{len(arms) * len(ablation_seeds)} cells, arm (a) reads the main run"} if reused else {}
    for stage in chosen:
        desc = ", ".join(filter(None, [STAGE_DESC.get(stage, ""), notes.get(stage)]))
        table.add_row(stage, str(counts.get(stage, 1)), desc)
    return table


def parse_args(argv=None):
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stages", nargs="+", default=DEFAULT_STAGES, choices=STAGES)
    p.add_argument("--aug", nargs="+", default=cfg["augmentations"], choices=cfg["augmentations"])
    p.add_argument("--seed", nargs="+", type=int, default=cfg["seeds"])
    p.add_argument("--epochs", type=int, default=None, help="default: phase1.epochs")
    p.add_argument("--limit", type=int, default=0, help="truncate the splits (smoke test)")
    p.add_argument("--ablation-arms", nargs="+", default=ABLATION_ARMS, choices=ABLATION_ARMS)
    p.add_argument("--ablation-seed", nargs="+", type=int, default=cfg["ablation_seeds"])
    p.add_argument("--xai-seed", type=int, default=None, help="default: the first seed")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--force", action="store_true", help="re-run cells whose artifacts already exist")
    p.add_argument("--dry-run", action="store_true", help="print the plan and the per-cell commands, run nothing")
    p.add_argument("--continue-on-error", action="store_true", help="keep going after a failed cell")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    cfg = load_config()
    if a.limit and not a.dry_run:
        refuse_real_results_dir(cfg["results_dir"])
    console = get_console()
    if not a.dry_run:
        check_fingerprint(cfg, console)
        check_adopted(cfg)
    xai_seed = a.xai_seed if a.xai_seed is not None else a.seed[0]

    mode = ["dry run"] * a.dry_run + ["force"] * a.force + ["continue on error"] * a.continue_on_error
    print_header_panel(
        title="MorphoMix",
        subtitle="",
        info_dict={
            "arms": ", ".join(a.aug),
            "seeds": ", ".join(str(s) for s in a.seed),
            "epochs": a.epochs or cfg["phase1"]["epochs"],
            **({"mode": ", ".join(mode)} if mode else {}),
        }
    )
    console.print(plan_table(cfg, a.stages, a.aug, a.seed, a.ablation_arms, a.ablation_seed))

    run = Runner(cfg, force=a.force, dry_run=a.dry_run, keep_going=a.continue_on_error)
    try:
        for stage in STAGES:
            if stage not in a.stages:
                continue
            run.rule(stage)
            if stage == "prepare":
                stage_prepare(run)
            elif stage == "phase1":
                stage_phase1(run, a.aug, a.seed, a.epochs, a.limit)
            elif stage == "foundation":
                stage_foundation(run, a.limit)
            elif stage == "tables":
                stage_tables(run, a.aug, a.seed)
            elif stage == "ablation":
                stage_ablation(run, a.ablation_arms, a.ablation_seed, a.epochs, a.limit)
            elif stage == "xai":
                stage_xai(run, a.aug, xai_seed, a.limit)
            elif stage == "stats":
                stage_stats(run, a.n_boot)
            elif stage == "analysis":
                stage_analysis(run, min(a.n_boot, 2000))
            elif stage == "shortcut":
                stage_shortcut(run)
            elif stage == "stress":
                stage_stress(run, a.limit)
            elif stage == "figures":
                stage_figures(run)
            elif stage == "profile":
                stage_profile(run)
    except KeyboardInterrupt:
        console.print("\n[warning]interrupted; run the same command again to resume[/warning]")
        run.summary()
        return 130
    except Exception:
        run.summary()
        raise

    if a.dry_run:
        console.print("\n[dim]dry run, nothing executed[/dim]")
        return 0
    return run.summary()


if __name__ == "__main__":
    main()
