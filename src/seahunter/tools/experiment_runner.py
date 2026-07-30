"""Validate, expand, and optionally execute detector experiment matrices."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from seahunter.experiments import DetectorExperimentRun, load_detector_experiment_suite
from seahunter.runtime import activate_legacy_ultralytics


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs/experiments/m0_detector_nwd_ablation.json",
    )
    parser.add_argument("--variant", action="append", help="run only the selected variant ID; repeatable")
    parser.add_argument("--device", default="0")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the expanded plan without training")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = _repo_root()
    suite = load_detector_experiment_suite(args.config, repo_root=root)
    selected = _select_runs(suite.runs, args.variant)
    plan = suite.to_record(root=root)
    plan["selected_run_count"] = len(selected)
    plan["selected_runs"] = [run.to_record(root=root) for run in selected]
    plan["dry_run"] = bool(args.dry_run)
    print(json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2))
    if args.dry_run:
        return 0

    ultralytics: Any = activate_legacy_ultralytics(root)
    for run in selected:
        _execute_run(run, ultralytics=ultralytics, device=args.device, suite_sha256=suite.source_sha256, root=root)
    return 0


def _select_runs(
    runs: tuple[DetectorExperimentRun, ...],
    selected_variants: list[str] | None,
) -> tuple[DetectorExperimentRun, ...]:
    if not selected_variants:
        return runs
    requested = set(selected_variants)
    available = {run.variant_id for run in runs}
    unknown = sorted(requested - available)
    if unknown:
        raise SystemExit(f"unknown experiment variants: {', '.join(unknown)}")
    return tuple(run for run in runs if run.variant_id in requested)


def _execute_run(
    run: DetectorExperimentRun,
    *,
    ultralytics: Any,
    device: str,
    suite_sha256: str,
    root: Path,
) -> None:
    output_dir = run.project / run.run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "experiment-manifest.json"
    manifest = {
        "schema_version": 1,
        "status": "started",
        "git_commit": os.environ.get("GITHUB_SHA") or _git_commit(root),
        "suite_sha256": suite_sha256,
        "python": platform.python_version(),
        "ultralytics": str(getattr(ultralytics, "__version__", "unknown")),
        "device": device,
        "run": run.to_record(root=root),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    model = ultralytics.YOLO(str(run.model))
    if run.initialization_weights is not None:
        model.load(str(run.initialization_weights))
    model.train(
        data=str(run.data),
        epochs=run.epochs,
        imgsz=run.image_size,
        batch=run.batch_size,
        workers=run.workers,
        seed=run.seed,
        deterministic=True,
        device=device,
        project=str(run.project),
        name=run.run_id,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        mosaic=1.0,
        copy_paste=0.3,
        mixup=0.1,
        degrees=0.0,
        patience=30,
        exist_ok=True,
        amp=True,
        nwd_weight=run.nwd_weight,
        nwd_constant=run.nwd_constant,
    )


def _git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None
