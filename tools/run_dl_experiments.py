from __future__ import annotations

import argparse
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from ds003029_eda.experiments.common import load_preset_config
from ds003029_eda.experiments.dl import DLExperimentConfig, run_dl_experiment
from ds003029_eda.paths import get_paths, resolve_workspace_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deep-learning experiments on feature or raw fold tensors.")
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--config", default=None, help="Optional JSON config containing defaults/presets.")
    parser.add_argument("--preset", default=None, help="Preset name inside the config JSON.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--input-mode", choices=["channel", "raw", "both"], default=None)
    parser.add_argument("--artifact-subdir", default=None)
    parser.add_argument("--raw-artifact-subdir", default=None)
    parser.add_argument("--output-subdir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    parser.add_argument("--mixed-precision", choices=["auto", "none", "fp16", "bf16"], default=None)
    parser.add_argument("--raw-loading-strategy", choices=["auto", "export", "on_demand"], default=None)
    return parser


def _build_config(args: argparse.Namespace) -> DLExperimentConfig:
    payload = load_preset_config(args.config, args.preset) if args.config else {}
    if args.model is not None:
        payload["model_name"] = args.model
    if args.experiment_name is not None:
        payload["experiment_name"] = args.experiment_name
    if args.input_mode is not None:
        payload["input_mode"] = args.input_mode
    if args.artifact_subdir is not None:
        payload["artifact_subdir"] = args.artifact_subdir
    if args.raw_artifact_subdir is not None:
        payload["raw_artifact_subdir"] = args.raw_artifact_subdir
    if args.output_subdir is not None:
        payload["output_subdir"] = args.output_subdir
    if args.epochs is not None:
        payload["epochs"] = args.epochs
    if args.batch_size is not None:
        payload["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        payload["learning_rate"] = args.learning_rate
    if args.random_state is not None:
        payload["random_state"] = args.random_state
    if args.device is not None:
        payload["device"] = args.device
    if args.mixed_precision is not None:
        payload["mixed_precision"] = args.mixed_precision
    if args.raw_loading_strategy is not None:
        payload["raw_loading_strategy"] = args.raw_loading_strategy

    if "model_name" not in payload or "input_mode" not in payload:
        raise ValueError("Both model_name and input_mode are required. Provide CLI args or a config preset.")

    payload.setdefault("experiment_name", payload["model_name"])
    return DLExperimentConfig(**payload)


def main() -> int:
    args = build_parser().parse_args()
    config = _build_config(args)
    paths = get_paths(resolve_workspace_root(args.workspace_root))
    metrics_df, aggregate_df = run_dl_experiment(config, paths=paths)

    print(f"Workspace root: {paths.workspace.as_posix()}")
    print(f"Experiment: {config.experiment_name}")
    print(f"Folds: {len(metrics_df)}")
    print(aggregate_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())