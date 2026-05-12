from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelingConfig:
    boundary_margin_sec: float = 0.5


def label_window(
    t_mid_s: float,
    intervals: list[tuple[float, float]],
    *,
    margin_sec: float = 0.5,
) -> int:
    for onset_s, offset_s in intervals:
        if onset_s + margin_sec <= t_mid_s <= offset_s - margin_sec:
            return 1

    for onset_s, offset_s in intervals:
        onset_boundary = onset_s - margin_sec <= t_mid_s <= onset_s + margin_sec
        offset_boundary = offset_s - margin_sec <= t_mid_s <= offset_s + margin_sec
        if onset_boundary or offset_boundary:
            return -1
    return 0


def label_windows(
    midpoints_s: np.ndarray | list[float],
    intervals: list[tuple[float, float]],
    *,
    margin_sec: float = 0.5,
) -> np.ndarray:
    mids = np.asarray(midpoints_s, dtype=float)
    return np.array([label_window(float(t_mid_s), intervals, margin_sec=margin_sec) for t_mid_s in mids], dtype=np.int8)


def build_window_label_frame(
    start_s: np.ndarray | list[float],
    stop_s: np.ndarray | list[float],
    intervals: list[tuple[float, float]],
    *,
    margin_sec: float = 0.5,
) -> pd.DataFrame:
    starts = np.asarray(start_s, dtype=float)
    stops = np.asarray(stop_s, dtype=float)
    mids = (starts + stops) / 2.0
    labels = label_windows(mids, intervals, margin_sec=margin_sec)
    return pd.DataFrame(
        {
            "t_start_s": starts,
            "t_stop_s": stops,
            "t_mid_s": mids,
            "y": labels,
        }
    )