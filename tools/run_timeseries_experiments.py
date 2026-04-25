from __future__ import annotations

import argparse
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from ds003029_eda.experiments.common import load_preset_config
from ds003029_eda.experiments.timeseries import TimeSeriesExperimentConfig, run_timeseries_experiment
from ds003029_eda.paths import get_paths, resolve_workspace_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SARIMA/SARIMAX timeseries experiments on v2 tensor exports.")
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--config", default=None, help="Optional JSON config containing defaults/presets.")
    parser.add_argument("--preset", default=None, help="Preset name inside the config JSON.")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--target-feature", default=None)
    parser.add_argument("--exog", nargs="*", default=None)
    parser.add_argument("--artifact-subdir", default=None)
    parser.add_argument("--output-subdir", default=None)
    parser.add_argument("--test-fraction", type=float, default=None)
    parser.add_argument("--min-total-obs", type=int, default=None)
    parser.add_argument("--min-test-obs", type=int, default=None)
    return parser


def _build_config(args: argparse.Namespace) -> TimeSeriesExperimentConfig:
    payload = load_preset_config(args.config, args.preset) if args.config else {}
    if args.experiment_name is not None:
        payload["experiment_name"] = args.experiment_name
    if args.target_feature is not None:
        payload["target_feature"] = args.target_feature
    if args.exog is not None:
        payload["exogenous_features"] = tuple(args.exog)
    if args.artifact_subdir is not None:
        payload["artifact_subdir"] = args.artifact_subdir
    if args.output_subdir is not None:
        payload["output_subdir"] = args.output_subdir
    if args.test_fraction is not None:
        payload["test_fraction"] = args.test_fraction
    if args.min_total_obs is not None:
        payload["min_total_obs"] = args.min_total_obs
    if args.min_test_obs is not None:
        payload["min_test_obs"] = args.min_test_obs

    if "experiment_name" not in payload:
        raise ValueError("An experiment name is required. Provide --experiment-name or a config preset.")
    if "target_feature" not in payload:
        payload["target_feature"] = "agg_mean_rms"
    payload.setdefault("exogenous_features", ())
    return TimeSeriesExperimentConfig(**payload)


def main() -> int:
    args = build_parser().parse_args()
    config = _build_config(args)
    paths = get_paths(resolve_workspace_root(args.workspace_root))
    prep_manifest_df, metrics_df = run_timeseries_experiment(config, paths=paths)

    print(f"Workspace root: {paths.workspace.as_posix()}")
    print(f"Experiment: {config.experiment_name}")
    print(f"Prepared series: {len(prep_manifest_df)}")
    print(metrics_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())