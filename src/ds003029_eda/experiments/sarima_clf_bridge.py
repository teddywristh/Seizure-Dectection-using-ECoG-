from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .common import compute_binary_metrics


DEFAULT_THRESHOLD_SWEEP = tuple(np.linspace(0.1, 0.9, 17).tolist())


def _compute_binary_metrics_quietly(y_true: np.ndarray, y_score: np.ndarray, *, threshold: float) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Only one class is present in y_true.*")
        warnings.filterwarnings("ignore", message="No positive class found in y_true.*")
        return compute_binary_metrics(y_true, y_score, threshold=threshold)


def residual_to_anomaly_score(
    residuals: np.ndarray,
    *,
    method: str = "zscore",
) -> np.ndarray:
    """Map SARIMA residuals to a monotonic anomaly score in [0, 1]."""
    abs_residuals = np.abs(np.asarray(residuals, dtype=float))
    if abs_residuals.size == 0:
        return abs_residuals

    if method == "zscore":
        mean_value = float(abs_residuals.mean())
        std_value = float(abs_residuals.std(ddof=0))
        if std_value < 1e-12:
            z_score = np.zeros_like(abs_residuals, dtype=float)
        else:
            z_score = (abs_residuals - mean_value) / (std_value + 1e-12)
        z_score = np.clip(z_score, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-z_score))

    if method == "minmax":
        min_value = float(abs_residuals.min())
        max_value = float(abs_residuals.max())
        span = max_value - min_value
        if span < 1e-12:
            return np.zeros_like(abs_residuals, dtype=float)
        return (abs_residuals - min_value) / (span + 1e-12)

    raise ValueError(f"Unknown residual score method: {method}")


def _prepare_eval_frame(
    pred_df: pd.DataFrame,
    *,
    residual_col: str,
    label_col: str,
    split_col: str,
    eval_split: str | None,
    boundary_label: int,
) -> pd.DataFrame:
    if residual_col not in pred_df.columns:
        raise KeyError(f"Missing residual column '{residual_col}' in prediction frame.")
    if label_col not in pred_df.columns:
        raise KeyError(f"Missing label column '{label_col}' in prediction frame.")

    eval_df = pred_df.copy()
    if eval_split is not None and split_col in eval_df.columns:
        eval_df = eval_df.loc[eval_df[split_col].astype(str) == eval_split].copy()

    eval_df[residual_col] = pd.to_numeric(eval_df[residual_col], errors="coerce")
    eval_df[label_col] = pd.to_numeric(eval_df[label_col], errors="coerce")
    eval_df = eval_df.dropna(subset=[residual_col, label_col])
    eval_df = eval_df.loc[eval_df[label_col] != boundary_label].copy()
    eval_df[label_col] = eval_df[label_col].astype(int)
    eval_df = eval_df.loc[eval_df[label_col].isin((0, 1))].copy()
    return eval_df.reset_index(drop=True)


def _extract_series_id(pred_df: pd.DataFrame) -> str | None:
    if "series_id" not in pred_df.columns:
        return None
    series_values = pred_df["series_id"].dropna().astype(str)
    if series_values.empty:
        return None
    return str(series_values.iloc[0])


def compute_sarima_classification_metrics(
    pred_df: pd.DataFrame,
    *,
    residual_col: str = "sarima_residual",
    label_col: str = "y",
    split_col: str = "split",
    eval_split: str | None = "test",
    boundary_label: int = -1,
    score_method: str = "zscore",
    thresholds: list[float] | tuple[float, ...] | None = None,
) -> dict[str, float | str]:
    eval_df = _prepare_eval_frame(
        pred_df,
        residual_col=residual_col,
        label_col=label_col,
        split_col=split_col,
        eval_split=eval_split,
        boundary_label=boundary_label,
    )
    if eval_df.empty:
        raise ValueError("No valid rows remained after split and label filtering for SARIMA classification metrics.")

    y_true = eval_df[label_col].to_numpy(dtype=int)
    y_score = residual_to_anomaly_score(eval_df[residual_col].to_numpy(dtype=float), method=score_method)

    candidate_thresholds = [float(value) for value in (thresholds or DEFAULT_THRESHOLD_SWEEP)]
    if not candidate_thresholds:
        candidate_thresholds = [0.5]

    best_threshold = candidate_thresholds[0]
    best_f1 = float("-inf")

    for threshold in candidate_thresholds:
        metric_values = _compute_binary_metrics_quietly(y_true, y_score, threshold=threshold)
        f1_value = float(metric_values.get("f1", float("nan")))
        if np.isnan(f1_value):
            continue
        if f1_value > best_f1 + 1e-12:
            best_f1 = f1_value
            best_threshold = threshold
            continue
        if abs(f1_value - best_f1) <= 1e-12 and abs(threshold - 0.5) < abs(best_threshold - 0.5):
            best_threshold = threshold

    final_metrics = _compute_binary_metrics_quietly(y_true, y_score, threshold=best_threshold)
    final_metrics["score_method"] = score_method
    final_metrics["best_threshold"] = float(best_threshold)
    final_metrics["n_eval_rows"] = float(len(eval_df))
    final_metrics["eval_split"] = "all" if eval_split is None else str(eval_split)
    return final_metrics


def run_sarima_classification_eval(
    *,
    experiment_dir: Path,
    training_output_name: str = "sarima_results",
    score_method: str = "zscore",
    residual_col: str = "sarima_residual",
    label_col: str = "y",
    split_col: str = "split",
    eval_split: str | None = "test",
    thresholds: list[float] | tuple[float, ...] | None = None,
    expected_series_ids: set[str] | None = None,
    allow_empty_predictions: bool = True,
) -> pd.DataFrame:
    pred_dir = experiment_dir / training_output_name / "predictions"
    if not pred_dir.exists():
        raise FileNotFoundError(f"Missing prediction directory for SARIMA classification bridge: {pred_dir}")

    selected_files_by_series: dict[str, Path] = {}
    cached_frames: dict[Path, pd.DataFrame] = {}

    for pred_file in sorted(pred_dir.glob("*_sarima_predictions.csv")):
        pred_df = pd.read_csv(pred_file)
        cached_frames[pred_file] = pred_df

        series_id = _extract_series_id(pred_df)
        if expected_series_ids and series_id is not None and series_id not in expected_series_ids:
            continue

        dedupe_key = series_id or pred_file.stem
        previous = selected_files_by_series.get(dedupe_key)
        if previous is None or pred_file.stat().st_mtime >= previous.stat().st_mtime:
            selected_files_by_series[dedupe_key] = pred_file

    rows: list[dict[str, float | str]] = []
    for dedupe_key in sorted(selected_files_by_series):
        pred_file = selected_files_by_series[dedupe_key]
        pred_df = cached_frames[pred_file]
        run_metrics = compute_sarima_classification_metrics(
            pred_df,
            residual_col=residual_col,
            label_col=label_col,
            split_col=split_col,
            eval_split=eval_split,
            score_method=score_method,
            thresholds=thresholds,
        )
        run_metrics["run_id"] = pred_file.stem.replace("_sarima_predictions", "")
        series_id = _extract_series_id(pred_df)
        if series_id is not None:
            run_metrics["series_id"] = series_id
        rows.append(run_metrics)

    output_path = experiment_dir / "sarima_classification_metrics.csv"
    if not rows and allow_empty_predictions:
        empty_df = pd.DataFrame()
        empty_df.to_csv(output_path, index=False)
        return empty_df
    if not rows:
        raise RuntimeError(f"No SARIMA prediction CSV files were found in {pred_dir}")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_path, index=False)
    return metrics_df
