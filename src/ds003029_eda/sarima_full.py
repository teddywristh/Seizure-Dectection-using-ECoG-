from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ds003029_eda.paths import WorkspacePaths, get_paths


DEFAULT_FEATURE_SOURCES = (
    "ds003029_window_features_demo.csv",
    "ds003029_window_features_demo.pkl",
)

ID_COLUMNS = {
    "base",
    "series_id",
    "run_id",
    "run",
    "subject",
    "acq",
}
TIME_COLUMNS = {"t_mid_s", "t_start", "t_end"}
LABEL_COLUMNS = {"y"}
NON_EXOG_COLUMNS = ID_COLUMNS | TIME_COLUMNS | LABEL_COLUMNS


@dataclass(frozen=True)
class PreparedData:
    df: pd.DataFrame
    target_col: str
    time_col: str
    series_col: str
    exog_cols: list[str]


@dataclass(frozen=True)
class FitArtifacts:
    series_id: str
    n_obs: int
    cadence_s: float
    candidate_seasonal_periods: list[int]
    target_col: str
    exog_cols: list[str]
    selected_order: tuple[int, int, int]
    selected_seasonal_order: tuple[int, int, int, int]
    selected_seasonal_period_steps: int | None
    selected_seasonal_period_seconds: float | None
    baseline_order: tuple[int, int, int]
    baseline_aic: float | None
    sarima_aic: float | None
    baseline_bic: float | None
    sarima_bic: float | None
    mse_arima: float
    mse_sarima: float
    rmse_arima: float
    rmse_sarima: float
    mae_arima: float
    mae_sarima: float
    median_ae_arima: float
    median_ae_sarima: float
    mape_arima_pct: float | None
    mape_sarima_pct: float | None
    smape_arima_pct: float | None
    smape_sarima_pct: float | None
    r2_arima: float | None
    r2_sarima: float | None
    ljungbox_pvalue_arima: float | None
    ljungbox_pvalue_sarima: float | None
    status: str
    notes: str


def _pick_existing_feature_source(outputs_dir: Path, requested: str | Path | None) -> Path:
    if requested is not None:
        candidate = Path(requested)
        if not candidate.is_absolute():
            candidate = outputs_dir / candidate
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Feature file not found: {candidate}")

    for name in DEFAULT_FEATURE_SOURCES:
        candidate = outputs_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find a feature file in {outputs_dir}. Tried: {DEFAULT_FEATURE_SOURCES}"
    )


def _load_features_dataframe(feature_path: Path) -> pd.DataFrame:
    if feature_path.suffix.lower() == ".pkl":
        return pd.read_pickle(feature_path)
    return pd.read_csv(feature_path)


def _infer_target_col(df: pd.DataFrame) -> str:
    for col in ("rms", "rms_mean"):
        if col in df.columns:
            return col
    raise ValueError("Could not infer target column. Expected 'rms' or 'rms_mean'.")


def _infer_time_col(df: pd.DataFrame) -> str:
    for col in ("t_mid_s", "t_start", "t_end"):
        if col in df.columns:
            return col
    raise ValueError("Could not infer time column. Expected one of t_mid_s, t_start, t_end.")


def _build_series_id(df: pd.DataFrame) -> pd.Series:
    if "series_id" in df.columns:
        return df["series_id"].astype(str)
    if "run_id" in df.columns:
        return df["run_id"].astype(str)
    if "base" in df.columns:
        return df["base"].astype(str)
    if "run" in df.columns:
        return df["run"].astype(str)
    return pd.Series(["series_000"] * len(df), index=df.index, dtype="string")


def prepare_full_series_dataset(
    feature_path: str | Path | None = None,
    *,
    paths: WorkspacePaths | None = None,
    n_lags: int = 2,
) -> PreparedData:
    paths = paths or get_paths()
    resolved_feature_path = _pick_existing_feature_source(paths.outputs_dir, feature_path)
    raw_df = _load_features_dataframe(resolved_feature_path).copy()

    target_col = _infer_target_col(raw_df)
    time_col = _infer_time_col(raw_df)

    if target_col != "rms":
        raw_df = raw_df.rename(columns={target_col: "rms"})
        target_col = "rms"

    raw_df["series_id"] = _build_series_id(raw_df)
    raw_df[time_col] = pd.to_numeric(raw_df[time_col], errors="coerce")
    raw_df["rms"] = pd.to_numeric(raw_df["rms"], errors="coerce")

    numeric_cols = [
        col
        for col in raw_df.columns
        if col not in NON_EXOG_COLUMNS and pd.api.types.is_numeric_dtype(raw_df[col])
    ]
    exog_cols = [col for col in numeric_cols if col != "rms"]

    working = raw_df.sort_values(["series_id", time_col]).reset_index(drop=True)
    for lag in range(1, n_lags + 1):
        lag_col = f"rms_lag_{lag}"
        working[lag_col] = working.groupby("series_id")["rms"].shift(lag)
        exog_cols.append(lag_col)

    exog_cols = list(dict.fromkeys(exog_cols))
    required = ["series_id", time_col, "rms", *exog_cols]
    if "y" in working.columns:
        required.append("y")
    prepared = working[required].dropna().reset_index(drop=True)

    return PreparedData(
        df=prepared,
        target_col="rms",
        time_col=time_col,
        series_col="series_id",
        exog_cols=exog_cols,
    )


def _candidate_orders() -> list[tuple[int, int, int]]:
    return [
        (1, 0, 0),
        (0, 0, 1),
        (1, 0, 1),
        (2, 0, 1),
        (1, 1, 1),
    ]


def _candidate_seasonal_orders(periods: Iterable[int]) -> list[tuple[int, int, int, int]]:
    seasonal_orders = [(0, 0, 0, 0)]
    for period in periods:
        seasonal_orders.extend(
            [
                (1, 0, 0, period),
                (0, 0, 1, period),
                (1, 0, 1, period),
                (1, 1, 0, period),
            ]
        )
    return seasonal_orders


def _safe_ljungbox_pvalue(residuals: pd.Series, lag: int = 10) -> float | None:
    clean = pd.Series(residuals).dropna()
    if len(clean) < lag + 5:
        return None
    try:
        out = acorr_ljungbox(clean, lags=[lag], return_df=True)
        return float(out["lb_pvalue"].iloc[-1])
    except Exception:
        return None


def _rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def _mape_pct(y_true: pd.Series, y_pred: pd.Series) -> float | None:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(true)
    mask = denom > 1e-12
    if not np.any(mask):
        return None
    return float(np.mean(np.abs((true[mask] - pred[mask]) / denom[mask])) * 100.0)


def _smape_pct(y_true: pd.Series, y_pred: pd.Series) -> float | None:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(true) + np.abs(pred)
    mask = denom > 1e-12
    if not np.any(mask):
        return None
    return float(np.mean(2.0 * np.abs(true[mask] - pred[mask]) / denom[mask]) * 100.0)


def _safe_r2(y_true: pd.Series, y_pred: pd.Series) -> float | None:
    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return None


def _fit_arima_baseline(
    y: pd.Series,
    exog: pd.DataFrame,
    order: tuple[int, int, int] = (1, 0, 1),
):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        model = ARIMA(endog=y, exog=exog, order=order)
        return model.fit()


def _fit_best_sarima(
    y: pd.Series,
    exog: pd.DataFrame,
    seasonal_periods: Iterable[int],
):
    best_result = None
    best_spec: tuple[tuple[int, int, int], tuple[int, int, int, int]] | None = None

    for order in _candidate_orders():
        for seasonal_order in _candidate_seasonal_orders(seasonal_periods):
            period = seasonal_order[3]
            if period and len(y) < max(30, period * 3):
                continue
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    warnings.filterwarnings("ignore", category=UserWarning)
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    model = SARIMAX(
                        endog=y,
                        exog=exog,
                        order=order,
                        seasonal_order=seasonal_order,
                        trend="c",
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    result = model.fit(disp=False, maxiter=200)
            except Exception:
                continue

            if best_result is None or result.aic < best_result.aic:
                best_result = result
                best_spec = (order, seasonal_order)

    if best_result is None or best_spec is None:
        raise RuntimeError("No SARIMA specification converged for this series.")
    return best_result, best_spec[0], best_spec[1]


def _infer_seasonal_periods(series_len: int) -> list[int]:
    base_periods = [4, 6, 12, 24]
    return [period for period in base_periods if series_len >= period * 3]


def _safe_series_filename(series_id: str) -> str:
    text = str(series_id)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    stem = Path(text).stem or "series"
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
    cleaned = cleaned.strip("_") or "series"
    return f"{cleaned[:18]}_{digest}"


def _seasonal_period_seconds(
    seasonal_order: tuple[int, int, int, int],
    cadence_s: float,
) -> float | None:
    period_steps = seasonal_order[3]
    if period_steps <= 0 or cadence_s <= 0:
        return None
    return float(period_steps * cadence_s)


def _summary_metadata_block(
    *,
    series_id: str,
    cadence_s: float,
    candidate_seasonal_periods: list[int],
    selected_order: tuple[int, int, int],
    selected_seasonal_order: tuple[int, int, int, int],
) -> str:
    period_steps = selected_seasonal_order[3] if selected_seasonal_order[3] > 0 else None
    period_seconds = _seasonal_period_seconds(selected_seasonal_order, cadence_s)
    lines = [
        "Custom SARIMA metadata",
        f"series_id: {series_id}",
        f"cadence_s: {cadence_s}",
        f"candidate_seasonal_periods_steps: {candidate_seasonal_periods}",
        f"selected_order: {selected_order}",
        f"selected_seasonal_order: {selected_seasonal_order}",
        (
            "selected_seasonal_period_steps: none"
            if period_steps is None
            else f"selected_seasonal_period_steps: {period_steps}"
        ),
        (
            "selected_seasonal_period_seconds: none"
            if period_seconds is None
            else f"selected_seasonal_period_seconds: {period_seconds}"
        ),
        (
            "interpretation: seasonal periods are measured in time steps, "
            "not seizure counts"
        ),
        "",
    ]
    return "\n".join(lines)


def fit_full_dataset_models(
    prepared: PreparedData,
    *,
    output_dir: Path,
) -> tuple[pd.DataFrame, list[FitArtifacts]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    summaries_dir = output_dir / "summaries"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    all_predictions: list[pd.DataFrame] = []
    metrics: list[FitArtifacts] = []

    for series_id, group in prepared.df.groupby(prepared.series_col, sort=True):
        group = group.sort_values(prepared.time_col).reset_index(drop=True)
        cadence_s = float(group[prepared.time_col].diff().median() or 0.0)
        seasonal_periods = _infer_seasonal_periods(len(group))
        if len(group) < 25:
            metrics.append(
                FitArtifacts(
                    series_id=str(series_id),
                    n_obs=int(len(group)),
                    cadence_s=cadence_s,
                    candidate_seasonal_periods=seasonal_periods,
                    target_col=prepared.target_col,
                    exog_cols=prepared.exog_cols,
                    selected_order=(0, 0, 0),
                    selected_seasonal_order=(0, 0, 0, 0),
                    selected_seasonal_period_steps=None,
                    selected_seasonal_period_seconds=None,
                    baseline_order=(1, 0, 1),
                    baseline_aic=None,
                    sarima_aic=None,
                    baseline_bic=None,
                    sarima_bic=None,
                    mse_arima=float("nan"),
                    mse_sarima=float("nan"),
                    rmse_arima=float("nan"),
                    rmse_sarima=float("nan"),
                    mae_arima=float("nan"),
                    mae_sarima=float("nan"),
                    median_ae_arima=float("nan"),
                    median_ae_sarima=float("nan"),
                    mape_arima_pct=None,
                    mape_sarima_pct=None,
                    smape_arima_pct=None,
                    smape_sarima_pct=None,
                    r2_arima=None,
                    r2_sarima=None,
                    ljungbox_pvalue_arima=None,
                    ljungbox_pvalue_sarima=None,
                    status="skipped",
                    notes="Series shorter than 25 observations after lagging.",
                )
            )
            continue

        y = group[prepared.target_col]
        exog = group[prepared.exog_cols]

        baseline_result = _fit_arima_baseline(y, exog)
        sarima_result, best_order, best_seasonal_order = _fit_best_sarima(
            y,
            exog,
            seasonal_periods,
        )

        baseline_pred = pd.Series(baseline_result.predict(), index=group.index)
        sarima_pred = pd.Series(sarima_result.predict(), index=group.index)

        pred_df = group[[prepared.series_col, prepared.time_col, prepared.target_col]].copy()
        pred_df["arima_fitted"] = baseline_pred
        pred_df["sarima_fitted"] = sarima_pred
        pred_df["arima_residual"] = pred_df[prepared.target_col] - pred_df["arima_fitted"]
        pred_df["sarima_residual"] = pred_df[prepared.target_col] - pred_df["sarima_fitted"]
        if "y" in group.columns:
            pred_df["y"] = group["y"].values

        safe_name = _safe_series_filename(str(series_id))
        pred_df.to_csv(predictions_dir / f"{safe_name}_full_fit_predictions.csv", index=False)
        (summaries_dir / f"{safe_name}_sarima_summary.txt").write_text(
            _summary_metadata_block(
                series_id=str(series_id),
                cadence_s=cadence_s,
                candidate_seasonal_periods=seasonal_periods,
                selected_order=best_order,
                selected_seasonal_order=best_seasonal_order,
            )
            + sarima_result.summary().as_text(),
            encoding="utf-8",
        )
        (summaries_dir / f"{safe_name}_arima_summary.txt").write_text(
            baseline_result.summary().as_text(),
            encoding="utf-8",
        )

        all_predictions.append(pred_df)
        metrics.append(
            FitArtifacts(
                series_id=str(series_id),
                n_obs=int(len(group)),
                cadence_s=cadence_s,
                candidate_seasonal_periods=seasonal_periods,
                target_col=prepared.target_col,
                exog_cols=prepared.exog_cols,
                selected_order=best_order,
                selected_seasonal_order=best_seasonal_order,
                selected_seasonal_period_steps=(
                    best_seasonal_order[3] if best_seasonal_order[3] > 0 else None
                ),
                selected_seasonal_period_seconds=_seasonal_period_seconds(
                    best_seasonal_order,
                    cadence_s,
                ),
                baseline_order=(1, 0, 1),
                baseline_aic=float(baseline_result.aic),
                sarima_aic=float(sarima_result.aic),
                baseline_bic=float(baseline_result.bic),
                sarima_bic=float(sarima_result.bic),
                mse_arima=float(mean_squared_error(y, baseline_pred)),
                mse_sarima=float(mean_squared_error(y, sarima_pred)),
                rmse_arima=_rmse(y, baseline_pred),
                rmse_sarima=_rmse(y, sarima_pred),
                mae_arima=float(mean_absolute_error(y, baseline_pred)),
                mae_sarima=float(mean_absolute_error(y, sarima_pred)),
                median_ae_arima=float(median_absolute_error(y, baseline_pred)),
                median_ae_sarima=float(median_absolute_error(y, sarima_pred)),
                mape_arima_pct=_mape_pct(y, baseline_pred),
                mape_sarima_pct=_mape_pct(y, sarima_pred),
                smape_arima_pct=_smape_pct(y, baseline_pred),
                smape_sarima_pct=_smape_pct(y, sarima_pred),
                r2_arima=_safe_r2(y, baseline_pred),
                r2_sarima=_safe_r2(y, sarima_pred),
                ljungbox_pvalue_arima=_safe_ljungbox_pvalue(pred_df["arima_residual"]),
                ljungbox_pvalue_sarima=_safe_ljungbox_pvalue(pred_df["sarima_residual"]),
                status="ok",
                notes=(
                    "Full-sequence fit without train/test split. "
                    "Model selection uses the lowest AIC among a compact candidate grid."
                ),
            )
        )

    metrics_df = pd.DataFrame([asdict(item) for item in metrics])
    metrics_df.to_csv(output_dir / "sarima_full_metrics.csv", index=False)
    if all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_csv(
            output_dir / "sarima_full_all_predictions.csv",
            index=False,
        )
    (output_dir / "sarima_full_run_config.json").write_text(
        json.dumps(
            {
                "feature_columns": prepared.df.columns.tolist(),
                "target_col": prepared.target_col,
                "time_col": prepared.time_col,
                "series_col": prepared.series_col,
                "exog_cols": prepared.exog_cols,
                "candidate_seasonal_periods_base_steps": [4, 6, 12, 24],
                "seasonal_period_unit": "time steps",
                "seasonal_period_seconds_formula": "selected_seasonal_period_steps * cadence_s",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics_df, metrics


def run_full_sarima_pipeline(
    *,
    feature_path: str | Path | None = None,
    n_lags: int = 2,
    output_subdir: str = "sarima_full_case",
    paths: WorkspacePaths | None = None,
) -> tuple[PreparedData, pd.DataFrame]:
    paths = paths or get_paths()
    prepared = prepare_full_series_dataset(feature_path, paths=paths, n_lags=n_lags)
    output_dir = paths.outputs_dir / output_subdir
    metrics_df, _ = fit_full_dataset_models(prepared, output_dir=output_dir)
    return prepared, metrics_df


def _console_safe(text: object) -> str:
    return str(text).encode("ascii", "backslashreplace").decode("ascii")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run full-sequence SARIMA/SARIMAX fitting on the available feature file "
            "without an explicit train/test split."
        )
    )
    parser.add_argument(
        "--features",
        default=None,
        help=(
            "Optional feature file path. By default the runner searches in eda_outputs/ "
            "for ds003029_window_features_demo.csv or .pkl."
        ),
    )
    parser.add_argument(
        "--n-lags",
        type=int,
        default=2,
        help="Number of RMS lag regressors to add.",
    )
    parser.add_argument(
        "--output-subdir",
        default="sarima_full_case",
        help="Subdirectory inside eda_outputs/ for metrics, predictions, and summaries.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    paths = get_paths()
    prepared, metrics_df = run_full_sarima_pipeline(
        feature_path=args.features,
        n_lags=args.n_lags,
        output_subdir=args.output_subdir,
        paths=paths,
    )

    output_dir = paths.outputs_dir / args.output_subdir
    ok_df = metrics_df[metrics_df["status"] == "ok"].copy()

    print("=" * 72)
    print("SARIMA FULL CASE COMPLETED")
    print("=" * 72)
    print(f"Input rows after lagging: {len(prepared.df)}")
    print(f"Series count: {prepared.df[prepared.series_col].nunique()}")
    print(f"Output directory: {_console_safe(output_dir.as_posix())}")
    print()

    if ok_df.empty:
        print("No successful fits. Check the metrics CSV for details.")
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
    ]
    printable_df = ok_df[display_cols].copy()
    printable_df["series_id"] = printable_df["series_id"].map(_console_safe)
    print(_console_safe(printable_df.to_string(index=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
