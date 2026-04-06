from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ws = Path.cwd().resolve()
src_dir = (ws / "src" if (ws / "src").exists() else ws.parent / "src").resolve()
sys.path.insert(0, str(src_dir))

from ds003029_eda.paths import get_paths
from ds003029_eda.sarima_full import run_full_sarima_pipeline
from ds003029_eda.window_features_multirun import WindowingConfig, build_multirun_window_features


def _console_safe(text: object) -> str:
    return str(text).encode("ascii", "backslashreplace").decode("ascii")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train SARIMA/SARIMAX on each full available time series without an "
            "explicit train/test split. Outputs metrics, fitted values, and model summaries."
        )
    )
    parser.add_argument(
        "--features",
        default=None,
        help=(
            "Optional feature file path. Defaults to eda_outputs/ds003029_window_features_demo.csv "
            "or .pkl if present."
        ),
    )
    parser.add_argument(
        "--n-lags",
        type=int,
        default=2,
        help="Number of autoregressive lag features of RMS to add as exogenous regressors.",
    )
    parser.add_argument(
        "--output-subdir",
        default="sarima_full_case",
        help="Subdirectory inside eda_outputs/ where artifacts will be written.",
    )
    parser.add_argument(
        "--build-multirun-features",
        action="store_true",
        help="Build fresh full-duration multirun features before training SARIMA.",
    )
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--step-sec", type=float, default=1.0)
    parser.add_argument("--max-channels", type=int, default=16)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

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

    prepared, metrics_df = run_full_sarima_pipeline(
        feature_path=feature_path,
        n_lags=args.n_lags,
        output_subdir=args.output_subdir,
        paths=paths,
    )

    output_dir = paths.outputs_dir / args.output_subdir
    ok_df = metrics_df[metrics_df["status"] == "ok"].copy()
    display_output_dir = _console_safe(output_dir.as_posix())

    print("=" * 72)
    print("FULL-DATASET SARIMA TRAINING COMPLETED")
    print("=" * 72)
    print(f"Input rows after lagging: {len(prepared.df)}")
    print(f"Series count: {prepared.df[prepared.series_col].nunique()}")
    print(f"Output directory: {display_output_dir}")
    print()

    if ok_df.empty:
        print("No model was successfully fitted. Check the metrics CSV for details.")
        return 1

    display_cols = [
        "series_id",
        "n_obs",
        "selected_order",
        "selected_seasonal_order",
        "sarima_aic",
        "mse_sarima",
        "rmse_sarima",
        "mae_sarima",
        "mape_sarima_pct",
        "smape_sarima_pct",
        "r2_sarima",
        "ljungbox_pvalue_sarima",
    ]
    print("Successful SARIMA fits:")
    printable_df = ok_df[display_cols].copy()
    for col in printable_df.columns:
        if printable_df[col].dtype == "object":
            printable_df[col] = printable_df[col].map(_console_safe)
    print(_console_safe(printable_df.to_string(index=False)))
    print()

    summary = pd.DataFrame(
        [
            {
                "mean_rmse_arima": ok_df["rmse_arima"].mean(),
                "mean_rmse_sarima": ok_df["rmse_sarima"].mean(),
                "mean_mse_sarima": ok_df["mse_sarima"].mean(),
                "mean_mae_arima": ok_df["mae_arima"].mean(),
                "mean_mae_sarima": ok_df["mae_sarima"].mean(),
                "mean_mape_sarima_pct": ok_df["mape_sarima_pct"].dropna().mean(),
                "mean_smape_sarima_pct": ok_df["smape_sarima_pct"].dropna().mean(),
                "mean_r2_sarima": ok_df["r2_sarima"].dropna().mean(),
                "mean_aic_gap_arima_minus_sarima": (
                    ok_df["baseline_aic"] - ok_df["sarima_aic"]
                ).mean(),
            }
        ]
    )
    print("Aggregate comparison:")
    print(_console_safe(summary.to_string(index=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
