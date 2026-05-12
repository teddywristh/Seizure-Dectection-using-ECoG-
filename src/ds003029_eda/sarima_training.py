from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA

try:
    import ruptures as rpt
except ImportError:  # pragma: no cover - optional dependency
    rpt = None

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ds003029_eda.paths import WorkspacePaths, get_paths
else:
    from .paths import WorkspacePaths, get_paths


DEFAULT_FEATURE_SOURCES = (
    "ds003029_window_features_multirun_full.csv",
    "ds003029_window_features_demo.csv",
    "ds003029_window_features_demo.pkl",
)

TIME_COLUMNS = ("t_mid_s", "t_start", "t_end")


@dataclass(frozen=True)
class PreparedData:
    df: pd.DataFrame
    target_col: str
    time_col: str
    series_col: str
    target_feature_name: str
    exogenous_cols: tuple[str, ...]


@dataclass(frozen=True)
class FitArtifacts:
    series_id: str
    n_obs: int
    n_train: int
    n_test: int
    cadence_s: float
    target_col: str
    ictal_window_rate: float | None
    exogenous_cols: tuple[str, ...]
    candidate_orders: list[tuple[int, int, int]]
    candidate_seasonal_periods_steps: list[int]
    selected_order: tuple[int, int, int]
    selected_seasonal_order: tuple[int, int, int, int]
    selected_seasonal_period_steps: int | None
    selected_seasonal_period_seconds: float | None
    aic_train: float | None
    bic_train: float | None
    hqic_train: float | None
    mse_test: float
    rmse_test: float
    mae_test: float
    median_ae_test: float
    mape_test_pct: float | None
    smape_test_pct: float | None
    r2_test: float | None
    ljungbox_pvalue_train_residual: float | None
    changepoints_train: list[float]
    changepoints_test: list[float]
    anomaly_count_test: int
    # Binary classification metrics from anomaly flags vs ground-truth y labels
    # (Only populated when 'y' column is present in the data — Issue #2 fix)
    anomaly_precision: float | None
    anomaly_recall: float | None
    anomaly_f1: float | None
    anomaly_average_precision: float | None
    anomaly_roc_auc: float | None
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
    for col in TIME_COLUMNS:
        if col in df.columns:
            return col
    raise ValueError("Could not infer time column. Expected one of t_mid_s, t_start, t_end.")


def _build_series_id(df: pd.DataFrame) -> pd.Series:
    for col in ("series_id", "run_id", "base", "run"):
        if col in df.columns:
            return df[col].astype(str)
    return pd.Series(["series_000"] * len(df), index=df.index, dtype="string")


def prepare_sarima_dataset(
    feature_path: str | Path | None = None,
    *,
    paths: WorkspacePaths | None = None,
    target_feature_name: str | None = None,
    exogenous_cols: tuple[str, ...] = (),
) -> PreparedData:
    paths = paths or get_paths()
    resolved_feature_path = _pick_existing_feature_source(paths.outputs_dir, feature_path)
    raw_df = _load_features_dataframe(resolved_feature_path).copy()

    target_col = _infer_target_col(raw_df)
    time_col = _infer_time_col(raw_df)
    if target_feature_name and target_feature_name in raw_df.columns:
        target_col = target_feature_name
    if target_col != "rms":
        raw_df = raw_df.rename(columns={target_col: "rms"})
        target_col = "rms"

    raw_df["series_id"] = _build_series_id(raw_df)
    raw_df[time_col] = pd.to_numeric(raw_df[time_col], errors="coerce")
    raw_df["rms"] = pd.to_numeric(raw_df["rms"], errors="coerce")

    keep_cols = ["series_id", time_col, "rms"]
    if "y" in raw_df.columns:
        raw_df["y"] = pd.to_numeric(raw_df["y"], errors="coerce")
        keep_cols.append("y")
    for col in exogenous_cols:
        if col not in raw_df.columns:
            raise KeyError(f"Requested exogenous column '{col}' was not found in {resolved_feature_path}")
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")
        keep_cols.append(col)

    prepared = (
        raw_df[keep_cols]
        .dropna(subset=["series_id", time_col, "rms"])
        .sort_values(["series_id", time_col])
        .reset_index(drop=True)
    )

    return PreparedData(
        df=prepared,
        target_col=target_col,
        time_col=time_col,
        series_col="series_id",
        target_feature_name=target_feature_name or target_col,
        exogenous_cols=tuple(exogenous_cols),
    )


def _candidate_orders() -> list[tuple[int, int, int]]:
    return [(1, 0, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1), (2, 1, 1)]


def _candidate_seasonal_orders(periods: list[int]) -> list[tuple[int, int, int, int]]:
    seasonal_orders = [(0, 0, 0, 0)]
    for period in periods:
        seasonal_orders.extend(
            [(1, 0, 0, period), (0, 1, 1, period), (1, 0, 1, period), (1, 1, 0, period)]
        )
    return seasonal_orders


def _infer_seasonal_periods(series_len: int) -> list[int]:
    base_periods = [4, 6, 12, 24]
    return [period for period in base_periods if series_len >= period * 3]


def _split_series(group: pd.DataFrame, test_fraction: float, min_test_obs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_obs = len(group)
    proposed_test = max(min_test_obs, int(math.ceil(n_obs * test_fraction)))
    n_test = min(proposed_test, max(1, n_obs - min_test_obs))
    split_idx = n_obs - n_test
    return group.iloc[:split_idx].copy(), group.iloc[split_idx:].copy()


def _fit_best_sarima(
    y_train: pd.Series,
    seasonal_periods: list[int],
    exog_train: pd.DataFrame | None = None,
) -> tuple[object, tuple[int, int, int], tuple[int, int, int, int]]:
    best_result = None
    best_spec = None

    for order in _candidate_orders():
        for seasonal_order in _candidate_seasonal_orders(seasonal_periods):
            period = seasonal_order[3]
            if period > 0 and len(y_train) < max(30, period * 3):
                continue

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    warnings.filterwarnings("ignore", category=UserWarning)
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    model = ARIMA(
                        endog=y_train,
                        exog=exog_train,
                        order=order,
                        seasonal_order=seasonal_order,
                        trend="c",
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    result = model.fit()
            except Exception:
                continue

            if best_result is None or result.aic < best_result.aic:
                best_result = result
                best_spec = (order, seasonal_order)

    if best_result is None or best_spec is None:
        raise RuntimeError("No SARIMA specification converged for this series.")

    return best_result, best_spec[0], best_spec[1]


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


def _safe_ljungbox_pvalue(residuals: pd.Series, lag: int = 10) -> float | None:
    clean = pd.Series(residuals).dropna()
    if len(clean) < lag + 5:
        return None
    try:
        out = acorr_ljungbox(clean, lags=[lag], return_df=True)
        return float(out["lb_pvalue"].iloc[-1])
    except Exception:
        return None


def _detect_changepoints(values: pd.Series, times: pd.Series, penalty: float) -> list[float]:
    if rpt is None or len(values) < 10:
        return []
    signal = np.asarray(values, dtype=float).reshape(-1, 1)
    try:
        breakpoints = rpt.Pelt(model="rbf").fit_predict(signal, pen=penalty)
    except Exception:
        return []

    clean_times = pd.Series(times).reset_index(drop=True)
    changepoints: list[float] = []
    for breakpoint in breakpoints[:-1]:
        if 0 < breakpoint <= len(clean_times):
            changepoints.append(float(clean_times.iloc[breakpoint - 1]))
    return changepoints


def _flag_residual_anomalies(
    residuals: pd.Series,
    *,
    rolling_window: int = 12,
    z_threshold: float = 2.5,
) -> pd.Series:
    series = pd.Series(residuals, dtype=float)
    rolling_std = series.abs().rolling(window=rolling_window, min_periods=max(3, rolling_window // 2)).std()
    rolling_std = rolling_std.replace(0.0, np.nan)
    return (series.abs() > (z_threshold * rolling_std)).fillna(False)


def _compute_anomaly_binary_metrics(
    y_true: np.ndarray,
    anomaly_mask: np.ndarray,
) -> dict[str, float | None]:
    """Compute binary classification metrics treating anomaly flags as predictions.

    This resolves the cross-method comparability issue (Issue.md #1/#2):
    SARIMA anomaly detection is evaluated with the same binary metrics (precision,
    recall, F1, AP, ROC-AUC) as ML and DL classifiers.

    Returns a dict with keys: anomaly_precision, anomaly_recall, anomaly_f1,
    anomaly_average_precision, anomaly_roc_auc.  All values are None if y_true
    has fewer than 2 unique classes (degenerate test window).
    """
    y_true_int = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(anomaly_mask, dtype=int)
    unique_classes = np.unique(y_true_int)
    if len(unique_classes) < 2:
        return {
            "anomaly_precision": None,
            "anomaly_recall": None,
            "anomaly_f1": None,
            "anomaly_average_precision": None,
            "anomaly_roc_auc": None,
        }
    try:
        prec = float(precision_score(y_true_int, y_pred, zero_division=0))
        rec = float(recall_score(y_true_int, y_pred, zero_division=0))
        f1 = float(f1_score(y_true_int, y_pred, zero_division=0))
        # Use anomaly flag as a binary probability proxy (0/1) for ranking-based metrics
        ap = float(average_precision_score(y_true_int, y_pred))
        auc = float(roc_auc_score(y_true_int, y_pred))
    except Exception:
        prec = rec = f1 = ap = auc = None  # type: ignore[assignment]
    return {
        "anomaly_precision": prec,
        "anomaly_recall": rec,
        "anomaly_f1": f1,
        "anomaly_average_precision": ap,
        "anomaly_roc_auc": auc,
    }


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
    n_train: int,
    n_test: int,
    candidate_orders: list[tuple[int, int, int]],
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
        f"n_train: {n_train}",
        f"n_test: {n_test}",
        f"candidate_orders: {candidate_orders}",
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
        "",
    ]
    return "\n".join(lines)


def fit_sarima_models(
    prepared: PreparedData,
    *,
    output_dir: Path,
    min_total_obs: int = 36,
    min_test_obs: int = 12,
    test_fraction: float = 0.2,
    changepoint_penalty: float = 10.0,
    residual_z_threshold: float = 2.5,
) -> tuple[pd.DataFrame, list[FitArtifacts]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    summaries_dir = output_dir / "summaries"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    all_predictions: list[pd.DataFrame] = []
    metrics: list[FitArtifacts] = []
    candidate_orders = _candidate_orders()

    for series_id, group in prepared.df.groupby(prepared.series_col, sort=True):
        group = group.sort_values(prepared.time_col).reset_index(drop=True)
        cadence_series = group[prepared.time_col].diff().dropna()
        cadence_s = float(cadence_series.median()) if not cadence_series.empty else 0.0
        ictal_window_rate = float(group["y"].fillna(0).mean()) if "y" in group.columns else None
        exog_cols = tuple(col for col in prepared.exogenous_cols if col in group.columns)

        if len(group) < min_total_obs:
            metrics.append(
                FitArtifacts(
                    series_id=str(series_id),
                    n_obs=int(len(group)),
                    n_train=0,
                    n_test=0,
                    cadence_s=cadence_s,
                    target_col=prepared.target_col,
                    ictal_window_rate=ictal_window_rate,
                    exogenous_cols=exog_cols,
                    candidate_orders=candidate_orders,
                    candidate_seasonal_periods_steps=[],
                    selected_order=(0, 0, 0),
                    selected_seasonal_order=(0, 0, 0, 0),
                    selected_seasonal_period_steps=None,
                    selected_seasonal_period_seconds=None,
                    aic_train=None,
                    bic_train=None,
                    hqic_train=None,
                    mse_test=float("nan"),
                    rmse_test=float("nan"),
                    mae_test=float("nan"),
                    median_ae_test=float("nan"),
                    mape_test_pct=None,
                    smape_test_pct=None,
                    r2_test=None,
                    ljungbox_pvalue_train_residual=None,
                    changepoints_train=[],
                    changepoints_test=[],
                    anomaly_count_test=0,
                    anomaly_precision=None,
                    anomaly_recall=None,
                    anomaly_f1=None,
                    anomaly_average_precision=None,
                    anomaly_roc_auc=None,
                    status="skipped",
                    notes=f"Series shorter than {min_total_obs} observations.",
                )
            )
            continue

        train_df, test_df = _split_series(group, test_fraction=test_fraction, min_test_obs=min_test_obs)
        if len(train_df) < min_total_obs - min_test_obs or len(test_df) < min_test_obs:
            metrics.append(
                FitArtifacts(
                    series_id=str(series_id),
                    n_obs=int(len(group)),
                    n_train=int(len(train_df)),
                    n_test=int(len(test_df)),
                    cadence_s=cadence_s,
                    target_col=prepared.target_col,
                    ictal_window_rate=ictal_window_rate,
                    exogenous_cols=exog_cols,
                    candidate_orders=candidate_orders,
                    candidate_seasonal_periods_steps=[],
                    selected_order=(0, 0, 0),
                    selected_seasonal_order=(0, 0, 0, 0),
                    selected_seasonal_period_steps=None,
                    selected_seasonal_period_seconds=None,
                    aic_train=None,
                    bic_train=None,
                    hqic_train=None,
                    mse_test=float("nan"),
                    rmse_test=float("nan"),
                    mae_test=float("nan"),
                    median_ae_test=float("nan"),
                    mape_test_pct=None,
                    smape_test_pct=None,
                    r2_test=None,
                    ljungbox_pvalue_train_residual=None,
                    changepoints_train=[],
                    changepoints_test=[],
                    anomaly_count_test=0,
                    anomaly_precision=None,
                    anomaly_recall=None,
                    anomaly_f1=None,
                    anomaly_average_precision=None,
                    anomaly_roc_auc=None,
                    status="skipped",
                    notes="Could not form a stable chronological train/test split.",
                )
            )
            continue

        y_train = pd.to_numeric(train_df[prepared.target_col], errors="coerce")
        y_test = pd.to_numeric(test_df[prepared.target_col], errors="coerce")
        exog_train = train_df[list(exog_cols)].astype(float) if exog_cols else None
        exog_test = test_df[list(exog_cols)].astype(float) if exog_cols else None
        seasonal_periods = _infer_seasonal_periods(len(train_df))

        result, best_order, best_seasonal_order = _fit_best_sarima(y_train, seasonal_periods, exog_train=exog_train)
        train_fitted = pd.Series(result.predict(start=0, end=len(train_df) - 1), index=train_df.index)
        test_forecast = pd.Series(result.forecast(steps=len(test_df), exog=exog_test), index=test_df.index)
        train_residual = y_train - train_fitted
        test_residual = y_test - test_forecast
        train_changepoints = _detect_changepoints(train_residual, train_df[prepared.time_col], changepoint_penalty)
        test_changepoints = _detect_changepoints(test_residual, test_df[prepared.time_col], changepoint_penalty)
        test_anomaly_mask = _flag_residual_anomalies(test_residual, z_threshold=residual_z_threshold)

        train_pred_df = train_df[[prepared.series_col, prepared.time_col, prepared.target_col]].copy()
        train_pred_df["split"] = "train"
        train_pred_df["sarima_pred"] = train_fitted
        train_pred_df["sarima_residual"] = train_residual
        train_pred_df["changepoint"] = train_pred_df[prepared.time_col].isin(train_changepoints)
        train_pred_df["anomaly"] = False

        test_pred_df = test_df[[prepared.series_col, prepared.time_col, prepared.target_col]].copy()
        test_pred_df["split"] = "test"
        test_pred_df["sarima_pred"] = test_forecast
        test_pred_df["sarima_residual"] = test_residual
        test_pred_df["changepoint"] = test_pred_df[prepared.time_col].isin(test_changepoints)
        test_pred_df["anomaly"] = test_anomaly_mask.values

        if exog_cols:
            train_pred_df = train_pred_df.join(train_df[list(exog_cols)].reset_index(drop=True))
            test_pred_df = test_pred_df.join(test_df[list(exog_cols)].reset_index(drop=True))

        pred_df = pd.concat([train_pred_df, test_pred_df], ignore_index=True)
        if "y" in group.columns:
            pred_df["y"] = group["y"].values

        safe_name = _safe_series_filename(str(series_id))
        pred_df.to_csv(predictions_dir / f"{safe_name}_sarima_predictions.csv", index=False)
        (summaries_dir / f"{safe_name}_sarima_summary.txt").write_text(
            _summary_metadata_block(
                series_id=str(series_id),
                cadence_s=cadence_s,
                n_train=int(len(train_df)),
                n_test=int(len(test_df)),
                candidate_orders=candidate_orders,
                candidate_seasonal_periods=seasonal_periods,
                selected_order=best_order,
                selected_seasonal_order=best_seasonal_order,
            )
            + result.summary().as_text(),
            encoding="utf-8",
        )

        all_predictions.append(pred_df)

        # Compute anomaly binary classification metrics vs ground-truth y (Issue #2 fix)
        anomaly_binary_metrics: dict[str, float | None] = {
            "anomaly_precision": None,
            "anomaly_recall": None,
            "anomaly_f1": None,
            "anomaly_average_precision": None,
            "anomaly_roc_auc": None,
        }
        if "y" in test_df.columns:
            y_test_labels = pd.to_numeric(test_df["y"], errors="coerce").fillna(0).astype(int).to_numpy()
            anomaly_binary_metrics = _compute_anomaly_binary_metrics(
                y_test_labels, test_anomaly_mask.to_numpy().astype(int)
            )

        metrics.append(
            FitArtifacts(
                series_id=str(series_id),
                n_obs=int(len(group)),
                n_train=int(len(train_df)),
                n_test=int(len(test_df)),
                cadence_s=cadence_s,
                target_col=prepared.target_col,
                ictal_window_rate=ictal_window_rate,
                exogenous_cols=exog_cols,
                candidate_orders=candidate_orders,
                candidate_seasonal_periods_steps=seasonal_periods,
                selected_order=best_order,
                selected_seasonal_order=best_seasonal_order,
                selected_seasonal_period_steps=(
                    best_seasonal_order[3] if best_seasonal_order[3] > 0 else None
                ),
                selected_seasonal_period_seconds=_seasonal_period_seconds(best_seasonal_order, cadence_s),
                aic_train=float(result.aic),
                bic_train=float(result.bic),
                hqic_train=float(result.hqic),
                mse_test=float(mean_squared_error(y_test, test_forecast)),
                rmse_test=_rmse(y_test, test_forecast),
                mae_test=float(mean_absolute_error(y_test, test_forecast)),
                median_ae_test=float(median_absolute_error(y_test, test_forecast)),
                mape_test_pct=_mape_pct(y_test, test_forecast),
                smape_test_pct=_smape_pct(y_test, test_forecast),
                r2_test=_safe_r2(y_test, test_forecast),
                ljungbox_pvalue_train_residual=_safe_ljungbox_pvalue(train_residual),
                changepoints_train=train_changepoints,
                changepoints_test=test_changepoints,
                anomaly_count_test=int(test_anomaly_mask.sum()),
                anomaly_precision=anomaly_binary_metrics["anomaly_precision"],
                anomaly_recall=anomaly_binary_metrics["anomaly_recall"],
                anomaly_f1=anomaly_binary_metrics["anomaly_f1"],
                anomaly_average_precision=anomaly_binary_metrics["anomaly_average_precision"],
                anomaly_roc_auc=anomaly_binary_metrics["anomaly_roc_auc"],
                status="ok",
                notes=(
                    "SARIMA/SARIMAX with chronological train/test split, residual changepoint detection, "
                    "and anomaly flagging."
                ),
            )
        )

    metrics_df = pd.DataFrame([asdict(item) for item in metrics])
    metrics_df.to_csv(output_dir / "sarima_metrics.csv", index=False)
    if all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_csv(
            output_dir / "sarima_all_predictions.csv",
            index=False,
        )

    (output_dir / "sarima_run_config.json").write_text(
        json.dumps(
            {
                "target_col": prepared.target_col,
                "target_feature_name": prepared.target_feature_name,
                "time_col": prepared.time_col,
                "series_col": prepared.series_col,
                "exogenous_cols": list(prepared.exogenous_cols),
                "default_feature_sources": list(DEFAULT_FEATURE_SOURCES),
                "candidate_orders": candidate_orders,
                "candidate_seasonal_periods_base_steps": [4, 6, 12, 24],
                "model_type": "SARIMAX" if prepared.exogenous_cols else "SARIMA",
                "uses_exog": bool(prepared.exogenous_cols),
                "changepoint": {
                    "method": "ruptures_pelt_rbf",
                    "enabled": rpt is not None,
                    "penalty": changepoint_penalty,
                },
                "anomaly_rule": {
                    "z_threshold": residual_z_threshold,
                    "signal": "test_residual_vs_rolling_std",
                },
                "evaluation": {
                    "scheme": "chronological_train_test_split",
                    "test_fraction": test_fraction,
                    "min_total_obs": min_total_obs,
                    "min_test_obs": min_test_obs,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics_df, metrics


def run_sarima_training(
    *,
    feature_path: str | Path | None = None,
    output_subdir: str = "sarima_training",
    paths: WorkspacePaths | None = None,
    min_total_obs: int = 36,
    min_test_obs: int = 12,
    test_fraction: float = 0.2,
    target_feature_name: str | None = None,
    exogenous_cols: tuple[str, ...] = (),
    changepoint_penalty: float = 10.0,
    residual_z_threshold: float = 2.5,
) -> tuple[PreparedData, pd.DataFrame]:
    paths = paths or get_paths()
    prepared = prepare_sarima_dataset(
        feature_path,
        paths=paths,
        target_feature_name=target_feature_name,
        exogenous_cols=exogenous_cols,
    )
    output_dir = paths.outputs_dir / output_subdir
    metrics_df, _ = fit_sarima_models(
        prepared,
        output_dir=output_dir,
        min_total_obs=min_total_obs,
        min_test_obs=min_test_obs,
        test_fraction=test_fraction,
        changepoint_penalty=changepoint_penalty,
        residual_z_threshold=residual_z_threshold,
    )
    return prepared, metrics_df


def _console_safe(text: object) -> str:
    return str(text).encode("ascii", "backslashreplace").decode("ascii")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pure SARIMA training with chronological train/test evaluation.")
    parser.add_argument("--features", default=None, help="Optional feature file path.")
    parser.add_argument("--output-subdir", default="sarima_training", help="Output subdirectory inside eda_outputs/.")
    parser.add_argument("--min-total-obs", type=int, default=36, help="Skip series shorter than this number of observations.")
    parser.add_argument("--min-test-obs", type=int, default=12, help="Minimum number of observations reserved for test.")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Chronological hold-out fraction for test.")
    parser.add_argument("--target-feature", default=None, help="Optional column name to use as the scalar SARIMA target.")
    parser.add_argument(
        "--exog",
        nargs="*",
        default=(),
        help="Optional exogenous columns already present in the SARIMA CSV, e.g. agg_std_rms y_lagged.",
    )
    parser.add_argument("--changepoint-penalty", type=float, default=10.0)
    parser.add_argument("--residual-z-threshold", type=float, default=2.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    paths = get_paths()
    prepared, metrics_df = run_sarima_training(
        feature_path=args.features,
        output_subdir=args.output_subdir,
        paths=paths,
        min_total_obs=args.min_total_obs,
        min_test_obs=args.min_test_obs,
        test_fraction=args.test_fraction,
        target_feature_name=args.target_feature,
        exogenous_cols=tuple(args.exog),
        changepoint_penalty=args.changepoint_penalty,
        residual_z_threshold=args.residual_z_threshold,
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
        print("No successful SARIMA fits. Check sarima_metrics.csv for details.")
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
    printable_df["series_id"] = printable_df["series_id"].map(_console_safe)
    print(_console_safe(printable_df.to_string(index=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
