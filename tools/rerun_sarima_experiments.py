"""Re-run all SARIMA experiments on the full dataset with fixed anomaly binary metrics.

This script reads from the existing sarima_v2_input.csv (already built by run_sarima_prep)
and re-runs SARIMA training for all presets, saving results with the new anomaly classification
metrics (Issue #2 fix). Results are saved in eda_outputs/experiments/timeseries/<preset>/.

Usage (from repo root, Seizure-Dectection-using-ECoG- dir):
    conda run -n drug-tox-env python tools/rerun_sarima_experiments.py \\
        --workspace-root /path/to/Seizure-Dectection-using-ECoG-
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

# Map of preset name -> config, matching configs/experiments/timeseries_models.json
PRESETS: dict[str, dict] = {
    "sarima_rms": {
        "target_feature": "rms",        # rms column in sarima_v2_input.csv
        "exogenous_cols": (),
    },
    "sarimax_rms_std": {
        "target_feature": "rms",
        "exogenous_cols": ("agg_std_rms",),   # column name in sarima_v2_input from sarimax runs
    },
    "sarimax_gamma": {
        "target_feature": "rms",
        "exogenous_cols": ("agg_std_gamma_high_power",),
    },
    "sarimax_hjorth": {
        "target_feature": "rms",
        "exogenous_cols": ("agg_std_hjorth_activity",),
    },
}

SARIMA_V2_INPUT_RELPATHS = [
    "data_processing_v2/sarima/ds003029_sarima_v2_input.csv",
    "data_processing_v2/sarima/sarima_rms/ds003029_sarima_v2_input.csv",
]


def _find_sarima_input(outputs_dir: Path, preset_name: str) -> Path | None:
    """Try to find the sarima_v2_input.csv for a given preset."""
    # Try preset-specific path first
    specific = outputs_dir / f"data_processing_v2/sarima/{preset_name}/ds003029_sarima_v2_input.csv"
    if specific.exists():
        return specific
    # Fall back to common combined CSV
    for relpath in SARIMA_V2_INPUT_RELPATHS:
        p = outputs_dir / relpath
        if p.exists():
            return p
    return None


def _save_experiment_output(
    metrics_df: pd.DataFrame,
    *,
    preset_name: str,
    outputs_dir: Path,
    output_subdir: str = "experiments/timeseries",
) -> Path:
    """Save series_metrics.csv (and run_config.json) in the standard experiment directory."""
    exp_dir = outputs_dir / output_subdir / preset_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(exp_dir / "series_metrics.csv", index=False)
    run_config = {
        "experiment_name": preset_name,
        "script": "tools/rerun_sarima_experiments.py",
        "note": "Re-run with anomaly binary classification metrics (Issue.md #1/#2 fix).",
    }
    (exp_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    return exp_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-run SARIMA experiments on the full dataset (sarima_v2_input.csv) with fixed anomaly binary metrics."
    )
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument(
        "--preset",
        default=None,
        choices=list(PRESETS.keys()),
        help="Run a single preset. Default: all.",
    )
    parser.add_argument("--output-subdir", default="experiments/timeseries")
    parser.add_argument("--min-total-obs", type=int, default=36)
    parser.add_argument("--min-test-obs", type=int, default=12)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--changepoint-penalty", type=float, default=10.0)
    parser.add_argument("--residual-z-threshold", type=float, default=2.5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = get_paths(resolve_workspace_root(args.workspace_root))

    print("=" * 72)
    print("SARIMA RE-RUN (with anomaly binary metrics — Issue #1/#2 fix)")
    print("=" * 72)
    print(f"Workspace: {paths.workspace.as_posix()}")
    print()

    presets_to_run = {args.preset: PRESETS[args.preset]} if args.preset else PRESETS
    success_count = 0
    fail_count = 0

    for preset_name, preset_cfg in presets_to_run.items():
        print(f"── Preset: {preset_name}")

        sarima_input = _find_sarima_input(paths.outputs_dir, preset_name)
        if sarima_input is None:
            print(f"   ✗ Could not find sarima_v2_input.csv for {preset_name}. Skipping.")
            fail_count += 1
            continue

        print(f"   Feature CSV: {sarima_input.name}")
        # Validate exogenous columns exist
        exog_cols = preset_cfg["exogenous_cols"]
        if exog_cols:
            df_check = pd.read_csv(sarima_input, nrows=1)
            missing_exog = [c for c in exog_cols if c not in df_check.columns]
            if missing_exog:
                print(f"   ⚠ Exogenous cols {missing_exog} not in CSV — running without exog.")
                exog_cols = tuple(c for c in exog_cols if c not in missing_exog)

        try:
            _, metrics_df = run_sarima_training(
                feature_path=sarima_input,
                output_subdir=f"sarima_training/{preset_name}",
                paths=paths,
                min_total_obs=args.min_total_obs,
                min_test_obs=args.min_test_obs,
                test_fraction=args.test_fraction,
                target_feature_name=preset_cfg["target_feature"],
                exogenous_cols=exog_cols,
                changepoint_penalty=args.changepoint_penalty,
                residual_z_threshold=args.residual_z_threshold,
            )
        except Exception as exc:
            print(f"   ✗ FAILED: {exc}")
            fail_count += 1
            continue

        ok_count = int((metrics_df["status"] == "ok").sum()) if "status" in metrics_df.columns else len(metrics_df)
        has_anomaly = "anomaly_f1" in metrics_df.columns
        anomaly_f1_vals = metrics_df["anomaly_f1"].dropna().tolist() if has_anomaly else []
        anomaly_roc_vals = metrics_df["anomaly_roc_auc"].dropna().tolist() if "anomaly_roc_auc" in metrics_df.columns else []

        print(f"   ✓ {ok_count}/{len(metrics_df)} series fitted | anomaly_f1={anomaly_f1_vals} | anomaly_roc_auc={anomaly_roc_vals}")

        exp_dir = _save_experiment_output(
            metrics_df,
            preset_name=preset_name,
            outputs_dir=paths.outputs_dir,
            output_subdir=args.output_subdir,
        )
        print(f"   → Saved: {exp_dir.as_posix()}")
        success_count += 1

    print()
    print(f"Completed: {success_count} presets OK, {fail_count} failed.")
    print("Run tools/summarize_experiment_metrics.py to regenerate summary.")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
