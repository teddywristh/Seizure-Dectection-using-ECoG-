from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from _workspace_cli import (
    append_optional_arg,
    load_config_payload,
    load_presets,
    resolve_preset_payload,
    resolve_repo_path,
    resolve_workspace_paths,
    run_python_script,
)


DEFAULT_CONFIGS = {
    "timeseries": "configs/experiments/timeseries_models.json",
    "ml": "configs/experiments/ml_models.json",
    "dl": "configs/experiments/dl_models.json",
    "all": "configs/experiments/all_models.json",
}


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--preset", default=None)
    parser.add_argument("--list-presets", action="store_true")
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Run training again even when a completed experiment output already exists.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Task 3: run experiment presets for timeseries, ML, or DL. "
            "Use --list-presets to inspect the canonical presets before training."
        )
    )
    subparsers = parser.add_subparsers(dest="family", required=True)

    timeseries_parser = subparsers.add_parser("timeseries", help="Run one timeseries preset.")
    _add_common_arguments(timeseries_parser)
    timeseries_parser.add_argument("--experiment-name", default=None)
    timeseries_parser.add_argument("--target-feature", default=None)

    ml_parser = subparsers.add_parser("ml", help="Run one machine-learning preset.")
    _add_common_arguments(ml_parser)
    ml_parser.add_argument("--experiment-name", default=None)
    ml_parser.add_argument("--threshold", type=float, default=None)
    ml_parser.add_argument("--optuna-trials", type=int, default=None)
    ml_parser.add_argument("--n-jobs", type=int, default=None)

    dl_parser = subparsers.add_parser("dl", help="Run one deep-learning preset.")
    _add_common_arguments(dl_parser)
    dl_parser.add_argument("--experiment-name", default=None)
    dl_parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    dl_parser.add_argument("--mixed-precision", choices=["auto", "none", "fp16", "bf16"], default=None)
    dl_parser.add_argument("--raw-loading-strategy", choices=["auto", "export", "on_demand"], default=None)
    dl_parser.add_argument("--epochs", type=int, default=None)
    dl_parser.add_argument("--batch-size", type=int, default=None)
    dl_parser.add_argument("--learning-rate", type=float, default=None)

    all_parser = subparsers.add_parser("all", help="Run all presets across timeseries, ML, and DL in grouped order.")
    _add_common_arguments(all_parser)
    return parser


def _resolve_config(args: argparse.Namespace) -> str:
    return resolve_repo_path(args.config or DEFAULT_CONFIGS[args.family]).as_posix()


def _print_presets(config_path: str) -> None:
    resolved_config, presets = load_presets(config_path)
    print(f"Preset config: {resolved_config.as_posix()}")
    for preset_name in sorted(str(name) for name in presets):
        print(preset_name)


def _load_all_config(config_path: str | Path) -> tuple[Path, list[str], dict[str, str]]:
    resolved_path, payload = load_config_payload(config_path)
    family_order = payload.get("family_order", ["timeseries", "ml", "dl"])
    family_configs = payload.get("family_configs", {})

    if not isinstance(family_order, list) or not all(isinstance(item, str) for item in family_order):
        raise ValueError(f"Invalid all-config at {resolved_path.as_posix()}: family_order must be a list of strings.")
    if not isinstance(family_configs, dict):
        raise ValueError(f"Invalid all-config at {resolved_path.as_posix()}: family_configs must be an object.")

    resolved_family_configs: dict[str, str] = {}
    for family in family_order:
        if family not in {"timeseries", "ml", "dl"}:
            raise ValueError(f"Unsupported family '{family}' in {resolved_path.as_posix()}.")
        family_config = family_configs.get(family, DEFAULT_CONFIGS[family])
        resolved_family_configs[family] = resolve_repo_path(str(family_config)).as_posix()
    return resolved_path, family_order, resolved_family_configs


def _print_all_presets(config_path: str) -> None:
    resolved_config, family_order, family_configs = _load_all_config(config_path)
    print(f"All config: {resolved_config.as_posix()}")
    for family in family_order:
        print(f"[{family}]")
        _family_config_path, presets = load_presets(family_configs[family])
        for preset_name in sorted(str(name) for name in presets):
            print(preset_name)


def _expected_experiment_dir(*, paths, family: str, preset_payload: dict[str, object], experiment_name_override: str | None) -> Path:
    experiment_name = experiment_name_override or str(preset_payload.get("experiment_name") or "")
    if not experiment_name:
        raise ValueError(f"Preset for family '{family}' does not define an experiment_name.")
    output_subdir = str(preset_payload.get("output_subdir") or f"experiments/{family}")
    return paths.outputs_dir / output_subdir / experiment_name


def _is_complete_experiment(family: str, experiment_dir: Path) -> bool:
    if family == "timeseries":
        required_files = (
            experiment_dir / "run_config.json",
            experiment_dir / "series_metrics.csv",
        )
        return all(path.exists() for path in required_files)

    required_files = [
        experiment_dir / "run_config.json",
        experiment_dir / "fold_metrics.csv",
        experiment_dir / "aggregate_metrics.csv",
        experiment_dir / "all_predictions.csv",
    ]
    if family == "dl":
        required_files.append(experiment_dir / "source_metadata.json")
    required_dirs = (
        experiment_dir / "predictions",
        experiment_dir / "checkpoints",
        experiment_dir / "reports",
    )
    return all(path.exists() for path in required_files) and all(path.exists() for path in required_dirs)


def _has_partial_artifacts(experiment_dir: Path) -> bool:
    return experiment_dir.exists() and any(experiment_dir.iterdir())


def _print_training_scope(*, paths, family: str) -> None:
    ready_manifest_path = paths.outputs_dir / "ds003029_model_ready_run_manifest.csv"
    n_ready_runs: int | None = None
    if ready_manifest_path.exists():
        n_ready_runs = int(len(pd.read_csv(ready_manifest_path)))

    if family == "timeseries":
        if n_ready_runs is not None:
            print(f"Data scope: {n_ready_runs} ready runs; SARIMA prep builds one time series per run for the selected preset.")
        else:
            print("Data scope: timeseries training uses the currently prepared SARIMA-ready run series.")
        return

    feature_inventory_path = paths.outputs_dir / "data_processing_v2" / "run_feature_inventory.csv"
    fold_manifest_path = paths.outputs_dir / "data_processing_v2" / "fold_manifest.json"
    if feature_inventory_path.exists():
        feature_inventory = pd.read_csv(feature_inventory_path)
        labeled_windows = int(
            pd.to_numeric(feature_inventory.get("n_ictal", 0), errors="coerce").fillna(0).sum()
            + pd.to_numeric(feature_inventory.get("n_interictal", 0), errors="coerce").fillna(0).sum()
        )
    else:
        labeled_windows = None

    n_folds: int | None = None
    if fold_manifest_path.exists():
        payload = json.loads(fold_manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            n_folds = len(payload)

    if n_ready_runs is not None and labeled_windows is not None and n_folds is not None:
        print(
            f"Data scope: {n_ready_runs} ready runs -> {labeled_windows} labeled windows split into {n_folds} LOSO folds."
        )
        return

    if n_ready_runs is not None:
        print(f"Data scope: {n_ready_runs} ready runs; ML/DL train on windows derived from those runs, not on 16 samples only.")
        return

    print("Data scope: ML/DL training uses the currently prepared fold windows, not raw run count as the sample count.")


def _handle_existing_outputs(*, family: str, experiment_dir: Path, force_retrain: bool) -> bool:
    if _is_complete_experiment(family, experiment_dir):
        if force_retrain:
            print(f"Completed outputs already exist at {experiment_dir.as_posix()}. Forcing retrain.")
            return False
        print(f"Using existing completed outputs: {experiment_dir.as_posix()}")
        print("No training was launched. Use --force-retrain if you want to retrain this preset.")
        return True

    if _has_partial_artifacts(experiment_dir):
        if force_retrain:
            print(f"Partial outputs detected at {experiment_dir.as_posix()}. Auto-resume is not implemented; retraining from scratch.")
            return False
        raise SystemExit(
            "Found partial existing outputs at "
            f"{experiment_dir.as_posix()}. Auto-resume from checkpoints is not implemented. "
            "Re-run with --force-retrain to retrain from scratch, or choose a new --experiment-name."
        )

    return False


def _build_command_for_family(args: argparse.Namespace, *, family: str, config_path: str, preset: str) -> list[str]:
    command = [
        "--workspace-root",
        args.workspace_root_resolved,
        "--config",
        config_path,
        "--preset",
        preset,
    ]

    if family == "timeseries":
        append_optional_arg(command, "--experiment-name", getattr(args, "experiment_name", None))
        append_optional_arg(command, "--target-feature", getattr(args, "target_feature", None))
        return command

    if family == "ml":
        append_optional_arg(command, "--experiment-name", getattr(args, "experiment_name", None))
        append_optional_arg(command, "--threshold", getattr(args, "threshold", None))
        append_optional_arg(command, "--optuna-trials", getattr(args, "optuna_trials", None))
        append_optional_arg(command, "--n-jobs", getattr(args, "n_jobs", None))
        return command

    append_optional_arg(command, "--experiment-name", getattr(args, "experiment_name", None))
    append_optional_arg(command, "--device", getattr(args, "device", None))
    append_optional_arg(command, "--mixed-precision", getattr(args, "mixed_precision", None))
    append_optional_arg(command, "--raw-loading-strategy", getattr(args, "raw_loading_strategy", None))
    append_optional_arg(command, "--epochs", getattr(args, "epochs", None))
    append_optional_arg(command, "--batch-size", getattr(args, "batch_size", None))
    append_optional_arg(command, "--learning-rate", getattr(args, "learning_rate", None))
    return command


def _run_single_preset(args: argparse.Namespace, *, family: str, config_path: str, preset: str, experiment_name_override: str | None = None) -> int:
    _, preset_payload = resolve_preset_payload(config_path, preset)
    paths = args.workspace_paths
    experiment_dir = _expected_experiment_dir(
        paths=paths,
        family=family,
        preset_payload=preset_payload,
        experiment_name_override=experiment_name_override,
    )

    _print_training_scope(paths=paths, family=family)
    if _handle_existing_outputs(
        family=family,
        experiment_dir=experiment_dir,
        force_retrain=args.force_retrain,
    ):
        return 0

    command = _build_command_for_family(args, family=family, config_path=config_path, preset=preset)
    if family == "timeseries":
        run_python_script("run_timeseries_experiments.py", *command)
        return 0
    if family == "ml":
        run_python_script("run_ml_experiments.py", *command)
        return 0
    run_python_script("run_dl_experiments.py", *command)
    return 0


def _run_all(args: argparse.Namespace) -> int:
    config_path = _resolve_config(args)
    _, family_order, family_configs = _load_all_config(config_path)
    for family in family_order:
        print(f"=== {family.upper()} ===")
        _, presets = load_presets(family_configs[family])
        for preset_name in sorted(str(name) for name in presets):
            print(f"Preset: {preset_name}")
            _run_single_preset(
                args,
                family=family,
                config_path=family_configs[family],
                preset=preset_name,
                experiment_name_override=None,
            )
    return 0


def main() -> int:
    args = build_parser().parse_args()
    args.workspace_paths = resolve_workspace_paths(args.workspace_root)
    args.workspace_root_resolved = args.workspace_paths.workspace.as_posix()

    if args.family == "all" and args.list_presets:
        _print_all_presets(_resolve_config(args))
        return 0
    if args.list_presets:
        _print_presets(_resolve_config(args))
        return 0
    if args.family == "all":
        return _run_all(args)
    if not args.preset:
        raise SystemExit("A preset is required. Use --preset <name> or --list-presets.")

    config_path = _resolve_config(args)
    return _run_single_preset(
        args,
        family=args.family,
        config_path=config_path,
        preset=args.preset,
        experiment_name_override=getattr(args, "experiment_name", None),
    )


if __name__ == "__main__":
    raise SystemExit(main())