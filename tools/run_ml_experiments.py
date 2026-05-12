from __future__ import annotations

import argparse
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from ds003029_eda.experiments.common import load_preset_config
from ds003029_eda.experiments.ml import MLExperimentConfig, run_ml_experiment
from ds003029_eda.paths import get_paths, resolve_workspace_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ML experiments on v2 LOSO fold artifacts.")
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--config", default=None, help="Optional JSON config containing defaults/presets.")
    parser.add_argument("--preset", default=None, help="Preset name inside the config JSON.")
    parser.add_argument("--model", default=None, help="Model name override.")
    parser.add_argument("--experiment-name", default=None, help="Output experiment name override.")
    parser.add_argument("--artifact-subdir", default=None)
    parser.add_argument("--output-subdir", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--optuna-trials", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    return parser


def _build_config(args: argparse.Namespace) -> MLExperimentConfig:
    payload = load_preset_config(args.config, args.preset) if args.config else {}
    if args.model is not None:
        payload["model_name"] = args.model
    if args.experiment_name is not None:
        payload["experiment_name"] = args.experiment_name
    if args.artifact_subdir is not None:
        payload["artifact_subdir"] = args.artifact_subdir
    if args.output_subdir is not None:
        payload["output_subdir"] = args.output_subdir
    if args.threshold is not None:
        payload["threshold"] = args.threshold
    if args.optuna_trials is not None:
        payload["optuna_trials"] = args.optuna_trials
    if args.n_jobs is not None:
        payload["n_jobs"] = args.n_jobs
    if args.random_state is not None:
        payload["random_state"] = args.random_state

    if "model_name" not in payload:
        raise ValueError("A model name is required. Provide --model or a config preset with model_name.")

    payload.setdefault("experiment_name", payload["model_name"])
    return MLExperimentConfig(**payload)


def main() -> int:
    args = build_parser().parse_args()
    config = _build_config(args)
    paths = get_paths(resolve_workspace_root(args.workspace_root))
    metrics_df, aggregate_df = run_ml_experiment(config, paths=paths)

    print(f"Workspace root: {paths.workspace.as_posix()}")
    print(f"Experiment: {config.experiment_name}")
    print(f"Folds: {len(metrics_df)}")
    print(aggregate_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())