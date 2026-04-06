from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ws = Path.cwd().resolve()
src_dir = (ws / "src" if (ws / "src").exists() else ws.parent / "src").resolve()
sys.path.insert(0, str(src_dir))

from ds003029_eda.paths import get_paths
from ds003029_eda.sarima_training import run_sarima_training
from ds003029_eda.window_features_multirun import WindowingConfig, build_multirun_window_features


def _console_safe(text: object) -> str:
    return str(text).encode("ascii", "backslashreplace").decode("ascii")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build multirun features if needed and train pure SARIMA models "
            "with chronological train/test evaluation."
        )
    )
    parser.add_argument("--features", default=None, help="Optional feature file path.")
    parser.add_argument("--output-subdir", default="sarima_training", help="Output subdirectory inside eda_outputs/.")
    parser.add_argument("--min-total-obs", type=int, default=36, help="Skip series shorter than this number of observations.")
    parser.add_argument("--min-test-obs", type=int, default=12, help="Minimum number of observations reserved for test.")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Chronological hold-out fraction for test.")
    parser.add_argument(
        "--build-multirun-features",
        action="store_true",
        help="Build fresh full-duration multirun features before SARIMA training.",
    )
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--step-sec", type=float, default=1.0)
    parser.add_argument("--max-channels", type=int, default=16)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = get_paths()
    feature_path = args.features

    if args.build_multirun_features:
        build_multirun_window_features(
            paths=paths,
            config=WindowingConfig(
                window_sec=args.window_sec,
                step_sec=args.step_sec,
                max_channels=args.max_channels,
            ),
            output_name="ds003029_window_features_multirun_full.csv",
            output_info_name="ds003029_windowing_multirun_full_info.csv",
        )
        feature_path = "ds003029_window_features_multirun_full.csv"

    prepared, metrics_df = run_sarima_training(
        feature_path=feature_path,
        output_subdir=args.output_subdir,
        paths=paths,
        min_total_obs=args.min_total_obs,
        min_test_obs=args.min_test_obs,
        test_fraction=args.test_fraction,
    )
    output_dir = paths.outputs_dir / args.output_subdir
    ok_df = metrics_df[metrics_df["status"] == "ok"].copy()

    print("=" * 72)
    print("SARIMA TRAINING COMPLETED")
    print("=" * 72)
    print(f"Input rows: {len(prepared.df)}")
    print(f"Series count: {prepared.df[prepared.series_col].nunique()}")
    print(f"Output directory: {_console_safe(output_dir.as_posix())}")
    print()

    if ok_df.empty:
        print("No model was successfully fitted. Check sarima_metrics.csv for details.")
        return 1

    display_cols = [
        "series_id",
        "n_train",
        "n_test",
        "selected_order",
        "selected_seasonal_order",
        "aic_train",
        "rmse_test",
        "mae_test",
        "mape_test_pct",
        "r2_test",
        "ljungbox_pvalue_train_residual",
    ]
    printable_df = ok_df[display_cols].copy()
    for col in printable_df.columns:
        if printable_df[col].dtype == "object":
            printable_df[col] = printable_df[col].map(_console_safe)
    print("Successful SARIMA fits:")
    print(_console_safe(printable_df.to_string(index=False)))
    print()

    summary = pd.DataFrame(
        [
            {
                "mean_aic_train": ok_df["aic_train"].mean(),
                "mean_rmse_test": ok_df["rmse_test"].mean(),
                "mean_mae_test": ok_df["mae_test"].mean(),
                "mean_mape_test_pct": ok_df["mape_test_pct"].dropna().mean(),
                "mean_r2_test": ok_df["r2_test"].dropna().mean(),
                "mean_ljungbox_pvalue_train_residual": ok_df["ljungbox_pvalue_train_residual"].dropna().mean(),
            }
        ]
    )
    print("Aggregate SARIMA summary:")
    print(_console_safe(summary.to_string(index=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
