"""Run SARIMA experiments directly from the demo/notebook feature CSV.

This script bypasses run_sarima_prep (which requires data_processing_v2 folds)
and runs SARIMA training directly using available demo features.
Results are saved in the standard experiments/timeseries/ structure so that
summarize_experiment_metrics.py can pick them up.

Usage:
    conda run -n drug-tox-env python tools/run_sarima_direct.py --workspace-root .
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

import pandas as pd

from ds003029_eda.paths import get_paths, resolve_workspace_root
from ds003029_eda.sarima_training import run_sarima_training

# Feature files to try in order
CANDIDATE_FEATURE_FILES = [
    "ds003029_window_features_multirun_full.csv",  # full dataset (if built)
    "ds003029_window_features_demo.csv",           # demo subset
    "ds003029_window_features_demo.pkl",           # demo subset (pkl)
]

# Notebook CSV with richer features (lives in notebooks/)
NOTEBOOK_CSV = repo_root / "notebooks" / "ds003029_arima_input_timeseries.csv"

PRESETS = {
    "sarima_rms": {
        "target_feature": "rms_mean",
        "exogenous_cols": (),
    },
    "sarimax_rms_hjorth": {
        "target_feature": "rms_mean",
        "exogenous_cols": ("hj_activity_mean",),
    },
}


def _pick_feature_file(outputs_dir: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = outputs_dir / explicit
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified feature file not found: {p}")

    # Try notebook CSV first (richer features)
    if NOTEBOOK_CSV.exists():
        return NOTEBOOK_CSV

    for name in CANDIDATE_FEATURE_FILES:
        candidate = outputs_dir / name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No feature CSV found. Tried {NOTEBOOK_CSV} and candidates in {outputs_dir}."
    )


def _save_experiment_metrics(
    metrics_df: pd.DataFrame,
    *,
    experiment_name: str,
    output_subdir: str,
    outputs_dir: Path,
) -> None:
    """Mirror a subset of metrics into experiments/timeseries/<name>/series_metrics.csv."""
    exp_dir = outputs_dir / output_subdir / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    # series_metrics.csv is what summarize_experiment_metrics.py reads
    metrics_df.to_csv(exp_dir / "series_metrics.csv", index=False)
    # Also write a minimal run_config.json for traceability
    run_config = {
        "experiment_name": experiment_name,
        "script": "tools/run_sarima_direct.py",
        "source": "direct SARIMA training from demo/notebook feature CSV",
    }
    (exp_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(f"  → Experiment output: {exp_dir.as_posix()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SARIMA experiments directly from available feature CSV, without needing data_processing_v2 folds."
    )
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--features", default=None, help="Optional explicit feature CSV path.")
    parser.add_argument("--output-subdir", default="experiments/timeseries")
    parser.add_argument("--preset", default=None, help="Run a single preset (sarima_rms | sarimax_rms_hjorth). Default: all.")
    parser.add_argument("--min-total-obs", type=int, default=30)
    parser.add_argument("--min-test-obs", type=int, default=8)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--changepoint-penalty", type=float, default=10.0)
    parser.add_argument("--residual-z-threshold", type=float, default=2.5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = get_paths(resolve_workspace_root(args.workspace_root))
    feature_path = _pick_feature_file(paths.outputs_dir, args.explicit_features if hasattr(args, "explicit_features") else args.features)

    print("=" * 72)
    print("SARIMA DIRECT EXPERIMENT RUNNER")
    print("=" * 72)
    print(f"Workspace root:  {paths.workspace.as_posix()}")
    print(f"Feature file:    {feature_path.as_posix()}")
    print()

    presets_to_run = {args.preset: PRESETS[args.preset]} if args.preset and args.preset in PRESETS else PRESETS

    for preset_name, preset_config in presets_to_run.items():
        target_feature = preset_config["target_feature"]
        exogenous_cols = preset_config["exogenous_cols"]

        print(f"Running preset: {preset_name}  (target={target_feature}, exog={exogenous_cols or 'none'})")

        try:
            _, metrics_df = run_sarima_training(
                feature_path=feature_path,
                output_subdir=f"sarima_training/{preset_name}",
                paths=paths,
                min_total_obs=args.min_total_obs,
                min_test_obs=args.min_test_obs,
                test_fraction=args.test_fraction,
                target_feature_name=target_feature,
                exogenous_cols=exogenous_cols,
                changepoint_penalty=args.changepoint_penalty,
                residual_z_threshold=args.residual_z_threshold,
            )
            ok_count = int((metrics_df["status"] == "ok").sum()) if "status" in metrics_df.columns else len(metrics_df)
            print(f"  → Fitted: {ok_count} / {len(metrics_df)} series")

            _save_experiment_metrics(
                metrics_df,
                experiment_name=preset_name,
                output_subdir=args.output_subdir,
                outputs_dir=paths.outputs_dir,
            )
        except Exception as exc:
            print(f"  ✗ Preset {preset_name} failed: {exc}")
            continue

    print()
    print("Done. Run tools/summarize_experiment_metrics.py to regenerate summary CSVs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
