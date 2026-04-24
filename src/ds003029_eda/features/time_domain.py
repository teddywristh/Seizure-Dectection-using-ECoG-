from __future__ import annotations

import numpy as np


TIME_DOMAIN_FEATURE_NAMES = (
    "rms",
    "line_length",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
    "zero_crossing_rate",
    "kurtosis",
    "skewness",
)

AGGREGATE_SUFFIXES = ("mean", "std", "max")


def _line_length(window: np.ndarray) -> np.ndarray:
    if window.shape[1] < 2:
        return np.zeros(window.shape[0], dtype=float)
    return np.sum(np.abs(np.diff(window, axis=1)), axis=1)


def _hjorth(window: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first_diff = np.diff(window, axis=1)
    second_diff = np.diff(first_diff, axis=1) if first_diff.shape[1] > 1 else np.zeros_like(first_diff)

    activity = np.var(window, axis=1)
    var_diff = np.var(first_diff, axis=1)
    var_diff2 = np.var(second_diff, axis=1) if second_diff.size else np.zeros(window.shape[0], dtype=float)

    mobility = np.sqrt(np.divide(var_diff, activity, out=np.zeros_like(var_diff), where=activity > 0))
    complexity = np.divide(
        np.sqrt(np.divide(var_diff2, var_diff, out=np.zeros_like(var_diff2), where=var_diff > 0)),
        mobility,
        out=np.zeros_like(mobility),
        where=mobility > 0,
    )
    return activity, mobility, complexity


def _zero_crossing_rate(window: np.ndarray) -> np.ndarray:
    centered = window - np.nanmean(window, axis=1, keepdims=True)
    signs = np.signbit(centered)
    if window.shape[1] < 2:
        return np.zeros(window.shape[0], dtype=float)
    return np.mean(signs[:, 1:] != signs[:, :-1], axis=1)


def _kurtosis(window: np.ndarray) -> np.ndarray:
    centered = window - np.nanmean(window, axis=1, keepdims=True)
    std = np.nanstd(centered, axis=1)
    fourth = np.nanmean(centered**4, axis=1)
    return np.divide(fourth, std**4, out=np.zeros_like(fourth), where=std > 0) - 3.0


def _skewness(window: np.ndarray) -> np.ndarray:
    centered = window - np.nanmean(window, axis=1, keepdims=True)
    std = np.nanstd(centered, axis=1)
    third = np.nanmean(centered**3, axis=1)
    return np.divide(third, std**3, out=np.zeros_like(third), where=std > 0)


def compute_time_domain_features(window: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(window, dtype=float)
    if values.ndim != 2:
        raise ValueError("Expected window with shape (channels, samples).")

    activity, mobility, complexity = _hjorth(values)
    return {
        "rms": np.sqrt(np.nanmean(values**2, axis=1)),
        "line_length": _line_length(values),
        "hjorth_activity": activity,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
        "zero_crossing_rate": _zero_crossing_rate(values),
        "kurtosis": _kurtosis(values),
        "skewness": _skewness(values),
    }