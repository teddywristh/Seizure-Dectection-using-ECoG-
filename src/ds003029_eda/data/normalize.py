from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ArrayScaler:
    feature_names: tuple[str, ...]
    mean_: tuple[float, ...]
    scale_: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "mean_": list(self.mean_),
            "scale_": list(self.scale_),
        }


def fit_array_scaler(
    data: np.ndarray,
    *,
    feature_names: list[str] | tuple[str, ...],
    valid_mask: np.ndarray | None = None,
) -> ArrayScaler:
    values = np.asarray(data, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim < 2:
        raise ValueError("Expected at least one feature dimension.")

    flat = values.reshape(-1, values.shape[-1])
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
        flat = flat[mask]

    if flat.size == 0:
        mean = np.zeros(values.shape[-1], dtype=float)
        scale = np.ones(values.shape[-1], dtype=float)
    else:
        mean = np.nanmean(flat, axis=0)
        scale = np.nanstd(flat, axis=0)
        mean = np.nan_to_num(mean, nan=0.0)
        scale = np.nan_to_num(scale, nan=1.0)
        scale[scale == 0.0] = 1.0

    return ArrayScaler(
        feature_names=tuple(str(name) for name in feature_names),
        mean_=tuple(float(v) for v in mean.tolist()),
        scale_=tuple(float(v) for v in scale.tolist()),
    )


def transform_array(
    data: np.ndarray,
    scaler: ArrayScaler,
    *,
    fill_missing_with: float | None = None,
) -> np.ndarray:
    values = np.asarray(data, dtype=float)
    mean = np.asarray(scaler.mean_, dtype=float)
    scale = np.asarray(scaler.scale_, dtype=float)
    transformed = (values - mean) / scale
    if fill_missing_with is not None:
        transformed = np.where(np.isfinite(transformed), transformed, float(fill_missing_with))
    return transformed


def save_scalers_json(path: Path, scalers: dict[str, ArrayScaler]) -> None:
    payload = {key: scaler.to_dict() for key, scaler in scalers.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")