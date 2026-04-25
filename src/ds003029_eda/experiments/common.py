from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from ..paths import WorkspacePaths


@dataclass(frozen=True)
class ExperimentDirectories:
    root: Path
    checkpoints_dir: Path
    predictions_dir: Path
    reports_dir: Path


def ensure_experiment_directories(
    paths: WorkspacePaths,
    *,
    output_subdir: str,
    experiment_name: str,
) -> ExperimentDirectories:
    root = paths.outputs_dir / output_subdir / experiment_name
    checkpoints_dir = root / "checkpoints"
    predictions_dir = root / "predictions"
    reports_dir = root / "reports"
    for directory in (root, checkpoints_dir, predictions_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return ExperimentDirectories(
        root=root,
        checkpoints_dir=checkpoints_dir,
        predictions_dir=predictions_dir,
        reports_dir=reports_dir,
    )


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_preset_config(path: str | Path, preset: str | None) -> dict[str, Any]:
    payload = load_json(path)
    if preset is None:
        return payload
    presets = payload.get("presets", {})
    if preset not in presets:
        raise KeyError(f"Preset '{preset}' was not found in {path}.")
    config = dict(payload.get("defaults", {}))
    config.update(presets[preset])
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _safe_metric(func, *args, **kwargs) -> float:
    try:
        return float(func(*args, **kwargs))
    except Exception:
        return float("nan")


def compute_binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else float("nan")

    return {
        "n_samples": float(y_true.size),
        "n_positive": float(np.sum(y_true == 1)),
        "n_negative": float(np.sum(y_true == 0)),
        "threshold": float(threshold),
        "roc_auc": _safe_metric(roc_auc_score, y_true, y_score),
        "average_precision": _safe_metric(average_precision_score, y_true, y_score),
        "f1": _safe_metric(f1_score, y_true, y_pred, zero_division=0),
        "precision": _safe_metric(precision_score, y_true, y_pred, zero_division=0),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": float(np.mean(y_pred == y_true)) if y_true.size else float("nan"),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def merge_prediction_frame(
    index_df: pd.DataFrame,
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    split: str,
) -> pd.DataFrame:
    frame = index_df.copy().reset_index(drop=True)
    frame["split"] = split
    frame["y_true"] = np.asarray(y_true, dtype=int)
    frame["y_score"] = np.asarray(y_score, dtype=float)
    frame["y_pred"] = (frame["y_score"] >= threshold).astype(int)
    return frame


def aggregate_metrics_frame(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()

    numeric_cols = [
        col
        for col in metrics_df.columns
        if col not in {"fold_id", "model_name", "experiment_name"} and pd.api.types.is_numeric_dtype(metrics_df[col])
    ]
    summary_rows: list[dict[str, float | str]] = []
    for column in numeric_cols:
        summary_rows.append(
            {
                "metric": column,
                "mean": float(metrics_df[column].mean()),
                "std": float(metrics_df[column].std(ddof=0)),
                "min": float(metrics_df[column].min()),
                "max": float(metrics_df[column].max()),
            }
        )
    return pd.DataFrame(summary_rows)