from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from ds003029_eda.paths import get_paths, resolve_workspace_root


STAGE_CHOICES = (
    "metadata",
    "data_processing",
    "raw_export",
    "timeseries",
    "ml",
    "dl",
    "summarize",
    "verify",
)

DEFAULT_STAGES = (
    "metadata",
    "data_processing",
    "timeseries",
    "ml",
    "dl",
    "summarize",
    "verify",
)


def _resolve_path(path_arg: str | None) -> Path | None:
    if path_arg is None:
        return None
    candidate = Path(path_arg)
    if candidate.is_absolute():
        return candidate.resolve()
    return (repo_root / candidate).resolve()


def _load_expected_experiments(config_path: Path) -> list[tuple[str, str]]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    defaults = payload.get("defaults", {})
    presets = payload.get("presets", {})
    if not isinstance(presets, dict):
        raise ValueError(f"Invalid config format in {config_path.as_posix()}: expected an object under 'presets'.")

    resolved: list[tuple[str, str]] = []
    for preset_name, preset_payload in sorted(presets.items()):
        merged = dict(defaults)
        merged.update(preset_payload)
        experiment_name = str(merged.get("experiment_name", preset_name))
        resolved.append((str(preset_name), experiment_name))
    return resolved


def _run_cmd(cmd: list[str], *, dry_run: bool) -> int:
    printable = " ".join(shlex.quote(token) for token in cmd)
    print(f"$ {printable}")
    if dry_run:
        return 0
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


def _is_timeseries_complete(experiment_dir: Path) -> bool:
    return (
        (experiment_dir / "run_config.json").exists()
        and (experiment_dir / "series_metrics.csv").exists()
    )


def _is_ml_complete(experiment_dir: Path) -> bool:
    required_files = (
        experiment_dir / "run_config.json",
        experiment_dir / "fold_metrics.csv",
        experiment_dir / "aggregate_metrics.csv",
        experiment_dir / "all_predictions.csv",
    )
    required_dirs = (
        experiment_dir / "predictions",
        experiment_dir / "checkpoints",
        experiment_dir / "reports",
    )
    return all(path.exists() for path in required_files) and all(path.exists() for path in required_dirs)


def _is_dl_complete(experiment_dir: Path) -> bool:
    required_files = (
        experiment_dir / "run_config.json",
        experiment_dir / "fold_metrics.csv",
        experiment_dir / "aggregate_metrics.csv",
        experiment_dir / "all_predictions.csv",
        experiment_dir / "source_metadata.json",
    )
    required_dirs = (
        experiment_dir / "predictions",
        experiment_dir / "checkpoints",
        experiment_dir / "reports",
    )
    return all(path.exists() for path in required_files) and all(path.exists() for path in required_dirs)


def _record_stage(
    records: list[dict[str, Any]],
    *,
    stage: str,
    item: str,
    status: str,
    command: list[str] | None = None,
    exit_code: int | None = None,
) -> None:
    records.append(
        {
            "stage": stage,
            "item": item,
            "status": status,
            "exit_code": exit_code,
            "command": command,
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
    )


def _execute_or_fail(
    records: list[dict[str, Any]],
    *,
    stage: str,
    item: str,
    command: list[str],
    dry_run: bool,
    continue_on_error: bool,
) -> bool:
    exit_code = _run_cmd(command, dry_run=dry_run)
    if exit_code == 0:
        _record_stage(records, stage=stage, item=item, status="completed", command=command, exit_code=0)
        return True

    _record_stage(records, stage=stage, item=item, status="failed", command=command, exit_code=exit_code)
    print(f"Command failed with exit code {exit_code}: {item}")
    return continue_on_error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the ds003029 workspace pipeline in a consistent order: metadata, data processing, "
            "timeseries, ML, DL, summarize, and verify."
        )
    )
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--python-executable", default=None, help="Python executable used to launch stage scripts.")
    parser.add_argument("--stages", nargs="+", choices=STAGE_CHOICES, default=list(DEFAULT_STAGES))
    parser.add_argument("--continue-on-error", action="store_true", help="Continue to the next command when one command fails.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only, without executing them.")

    parser.add_argument("--artifact-subdir", default="data_processing_v2")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.set_defaults(skip_existing=True)

    parser.add_argument("--timeseries-config", default="configs/experiments/timeseries_models.json")
    parser.add_argument("--ml-config", default="configs/experiments/ml_models.json")
    parser.add_argument("--dl-config", default="configs/experiments/dl_models.json")

    parser.add_argument("--target-sfreq", type=float, default=256.0)
    parser.add_argument("--bandpass-low", type=float, default=0.5)
    parser.add_argument("--bandpass-high", type=float, default=120.0)
    parser.add_argument("--notch-harmonics", type=int, default=3)
    parser.add_argument("--reference-mode", default="average")
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--step-sec", type=float, default=0.5)
    parser.add_argument("--label-margin-sec", type=float, default=0.5)
    parser.add_argument("--fill-missing-after-scaling", type=float, default=0.0)
    parser.add_argument("--overwrite-data-processing", action="store_true")

    parser.add_argument("--raw-output-subdir", default="raw_folds")
    parser.add_argument("--raw-window-sec", type=float, default=2.0)
    parser.add_argument("--raw-output-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--overwrite-raw-export", action="store_true")

    parser.add_argument("--dl-device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dl-mixed-precision", choices=("auto", "none", "fp16", "bf16"), default="auto")
    parser.add_argument("--dl-raw-loading-strategy", choices=("auto", "export", "on_demand"), default="auto")

    parser.add_argument("--summary-output-dir", default=None)
    parser.add_argument("--verification-output-dir", default=None)
    parser.add_argument("--verify-fail-on-error", dest="verify_fail_on_error", action="store_true")
    parser.add_argument("--verify-no-fail-on-error", dest="verify_fail_on_error", action="store_false")
    parser.set_defaults(verify_fail_on_error=True)

    return parser


def main() -> int:
    args = build_parser().parse_args()

    workspace_root = resolve_workspace_root(args.workspace_root)
    paths = get_paths(workspace_root)
    experiments_root = paths.outputs_dir / "experiments"

    python_executable = args.python_executable or sys.executable

    timeseries_config_path = _resolve_path(args.timeseries_config)
    ml_config_path = _resolve_path(args.ml_config)
    dl_config_path = _resolve_path(args.dl_config)

    if timeseries_config_path is None or not timeseries_config_path.exists():
        raise FileNotFoundError(f"Missing timeseries config: {timeseries_config_path}")
    if ml_config_path is None or not ml_config_path.exists():
        raise FileNotFoundError(f"Missing ML config: {ml_config_path}")
    if dl_config_path is None or not dl_config_path.exists():
        raise FileNotFoundError(f"Missing DL config: {dl_config_path}")

    stage_records: list[dict[str, Any]] = []

    print(f"Workspace root: {workspace_root.as_posix()}")
    print(f"Python executable: {python_executable}")
    print(f"Stages: {', '.join(args.stages)}")

    if "metadata" in args.stages:
        metadata_commands = [
            (
                "build_run_summary",
                [
                    python_executable,
                    str(repo_root / "tools" / "build_run_summary_ds003029.py"),
                    "--workspace-root",
                    workspace_root.as_posix(),
                ],
            ),
            (
                "build_marker_qc",
                [
                    python_executable,
                    str(repo_root / "tools" / "build_marker_qc_ds003029.py"),
                    "--workspace-root",
                    workspace_root.as_posix(),
                ],
            ),
        ]
        for item, command in metadata_commands:
            should_continue = _execute_or_fail(
                stage_records,
                stage="metadata",
                item=item,
                command=command,
                dry_run=args.dry_run,
                continue_on_error=args.continue_on_error,
            )
            if not should_continue:
                return 1

    if "data_processing" in args.stages:
        command = [
            python_executable,
            str(repo_root / "tools" / "run_data_processing_v2.py"),
            "full",
            "--workspace-root",
            workspace_root.as_posix(),
            "--artifact-subdir",
            args.artifact_subdir,
            "--target-sfreq",
            str(args.target_sfreq),
            "--bandpass-low",
            str(args.bandpass_low),
            "--bandpass-high",
            str(args.bandpass_high),
            "--notch-harmonics",
            str(args.notch_harmonics),
            "--reference-mode",
            args.reference_mode,
            "--window-sec",
            str(args.window_sec),
            "--step-sec",
            str(args.step_sec),
            "--label-margin-sec",
            str(args.label_margin_sec),
            "--fill-missing-after-scaling",
            str(args.fill_missing_after_scaling),
        ]
        if args.overwrite_data_processing:
            command.append("--overwrite")

        should_continue = _execute_or_fail(
            stage_records,
            stage="data_processing",
            item="run_data_processing_v2_full",
            command=command,
            dry_run=args.dry_run,
            continue_on_error=args.continue_on_error,
        )
        if not should_continue:
            return 1

    if "raw_export" in args.stages:
        command = [
            python_executable,
            str(repo_root / "tools" / "build_raw_folds.py"),
            "--workspace-root",
            workspace_root.as_posix(),
            "--artifact-subdir",
            args.artifact_subdir,
            "--output-subdir",
            args.raw_output_subdir,
            "--window-sec",
            str(args.raw_window_sec),
            "--output-dtype",
            args.raw_output_dtype,
        ]
        if args.overwrite_raw_export:
            command.append("--overwrite")

        should_continue = _execute_or_fail(
            stage_records,
            stage="raw_export",
            item="build_raw_folds",
            command=command,
            dry_run=args.dry_run,
            continue_on_error=args.continue_on_error,
        )
        if not should_continue:
            return 1

    if "timeseries" in args.stages:
        timeseries_presets = _load_expected_experiments(timeseries_config_path)
        for preset_name, experiment_name in timeseries_presets:
            experiment_dir = experiments_root / "timeseries" / experiment_name
            if args.skip_existing and _is_timeseries_complete(experiment_dir):
                print(f"Skip existing timeseries experiment: {experiment_name}")
                _record_stage(
                    stage_records,
                    stage="timeseries",
                    item=preset_name,
                    status="skipped",
                )
                continue

            command = [
                python_executable,
                str(repo_root / "tools" / "run_timeseries_experiments.py"),
                "--workspace-root",
                workspace_root.as_posix(),
                "--config",
                timeseries_config_path.as_posix(),
                "--preset",
                preset_name,
            ]
            should_continue = _execute_or_fail(
                stage_records,
                stage="timeseries",
                item=preset_name,
                command=command,
                dry_run=args.dry_run,
                continue_on_error=args.continue_on_error,
            )
            if not should_continue:
                return 1

    if "ml" in args.stages:
        ml_presets = _load_expected_experiments(ml_config_path)
        for preset_name, experiment_name in ml_presets:
            experiment_dir = experiments_root / "ml" / experiment_name
            if args.skip_existing and _is_ml_complete(experiment_dir):
                print(f"Skip existing ML experiment: {experiment_name}")
                _record_stage(
                    stage_records,
                    stage="ml",
                    item=preset_name,
                    status="skipped",
                )
                continue

            command = [
                python_executable,
                str(repo_root / "tools" / "run_ml_experiments.py"),
                "--workspace-root",
                workspace_root.as_posix(),
                "--config",
                ml_config_path.as_posix(),
                "--preset",
                preset_name,
            ]
            should_continue = _execute_or_fail(
                stage_records,
                stage="ml",
                item=preset_name,
                command=command,
                dry_run=args.dry_run,
                continue_on_error=args.continue_on_error,
            )
            if not should_continue:
                return 1

    if "dl" in args.stages:
        dl_presets = _load_expected_experiments(dl_config_path)
        for preset_name, experiment_name in dl_presets:
            experiment_dir = experiments_root / "dl" / experiment_name
            if args.skip_existing and _is_dl_complete(experiment_dir):
                print(f"Skip existing DL experiment: {experiment_name}")
                _record_stage(
                    stage_records,
                    stage="dl",
                    item=preset_name,
                    status="skipped",
                )
                continue

            command = [
                python_executable,
                str(repo_root / "tools" / "run_dl_experiments.py"),
                "--workspace-root",
                workspace_root.as_posix(),
                "--config",
                dl_config_path.as_posix(),
                "--preset",
                preset_name,
                "--device",
                args.dl_device,
                "--mixed-precision",
                args.dl_mixed_precision,
                "--raw-loading-strategy",
                args.dl_raw_loading_strategy,
            ]
            should_continue = _execute_or_fail(
                stage_records,
                stage="dl",
                item=preset_name,
                command=command,
                dry_run=args.dry_run,
                continue_on_error=args.continue_on_error,
            )
            if not should_continue:
                return 1

    if "summarize" in args.stages:
        command = [
            python_executable,
            str(repo_root / "tools" / "summarize_experiment_metrics.py"),
            "--workspace-root",
            workspace_root.as_posix(),
        ]
        if args.summary_output_dir is not None:
            command.extend(["--output-dir", str(Path(args.summary_output_dir).resolve())])

        should_continue = _execute_or_fail(
            stage_records,
            stage="summarize",
            item="summarize_experiment_metrics",
            command=command,
            dry_run=args.dry_run,
            continue_on_error=args.continue_on_error,
        )
        if not should_continue:
            return 1

    if "verify" in args.stages:
        command = [
            python_executable,
            str(repo_root / "tools" / "verify_experiment_outputs.py"),
            "--workspace-root",
            workspace_root.as_posix(),
        ]
        if args.verification_output_dir is not None:
            command.extend(["--output-dir", str(Path(args.verification_output_dir).resolve())])
        if args.verify_fail_on_error:
            command.append("--fail-on-error")
        else:
            command.append("--no-fail-on-error")

        should_continue = _execute_or_fail(
            stage_records,
            stage="verify",
            item="verify_experiment_outputs",
            command=command,
            dry_run=args.dry_run,
            continue_on_error=args.continue_on_error,
        )
        if not should_continue:
            return 1

    report_dir = experiments_root / "pipeline_runs"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"pipeline_run_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    report_path.write_text(
        json.dumps(
            {
                "workspace_root": workspace_root.as_posix(),
                "python_executable": python_executable,
                "stages": args.stages,
                "skip_existing": bool(args.skip_existing),
                "dry_run": bool(args.dry_run),
                "continue_on_error": bool(args.continue_on_error),
                "records": stage_records,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    failed_count = sum(1 for record in stage_records if record["status"] == "failed")
    skipped_count = sum(1 for record in stage_records if record["status"] == "skipped")
    completed_count = sum(1 for record in stage_records if record["status"] == "completed")

    print(f"Pipeline report: {report_path.as_posix()}")
    print(f"Completed: {completed_count}, skipped: {skipped_count}, failed: {failed_count}")

    return 1 if failed_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
