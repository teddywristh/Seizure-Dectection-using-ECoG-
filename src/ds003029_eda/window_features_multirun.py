from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ds003029_eda.paths import WorkspacePaths, get_paths


@dataclass(frozen=True)
class WindowingConfig:
    window_sec: float = 2.0
    step_sec: float = 1.0
    max_channels: int = 16


def _ensure_mne_workspace_home(paths: WorkspacePaths) -> Path:
    mne_home = paths.workspace / ".mne"
    mne_home.mkdir(parents=True, exist_ok=True)
    os.environ["_MNE_FAKE_HOME_DIR"] = str(paths.workspace)
    os.environ["MNE_HOME"] = str(mne_home)
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    return mne_home


def _load_mne(paths: WorkspacePaths):
    _ensure_mne_workspace_home(paths)
    import mne

    return mne


def _line_length(x: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(x))))


def _resolve_existing_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    alt = Path.cwd() / path
    if alt.exists():
        return alt
    return path


def _iter_target_runs(paths: WorkspacePaths) -> pd.DataFrame:
    run_summary = pd.read_csv(paths.outputs_dir / "ds003029_run_summary.csv")
    intervals = pd.read_csv(paths.outputs_dir / "ds003029_seizure_intervals_by_run.csv")

    interval_bases = set(intervals["base"].astype(str))
    if "eeg_content_present" not in run_summary.columns:
        raise KeyError("run_summary missing eeg_content_present column")

    runs = run_summary[run_summary["eeg_content_present"].astype(bool)].copy()
    runs["has_interval"] = runs["base"].astype(str).isin(interval_bases)
    runs = runs[runs["has_interval"]].copy()
    if runs.empty:
        raise RuntimeError("No runs with eeg content and paired seizure intervals were found.")
    return runs, intervals


def build_multirun_window_features(
    *,
    paths: WorkspacePaths | None = None,
    config: WindowingConfig | None = None,
    output_name: str = "ds003029_window_features_multirun_full.csv",
    output_info_name: str = "ds003029_windowing_multirun_full_info.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = paths or get_paths()
    config = config or WindowingConfig()
    mne = _load_mne(paths)
    runs, intervals = _iter_target_runs(paths)

    feature_rows: list[dict] = []
    info_rows: list[dict] = []

    for _, row in runs.sort_values("base").iterrows():
        base = str(row["base"])
        vhdr = _resolve_existing_path(base + ".vhdr")
        if not vhdr.exists():
            warnings.warn(f"Skipping run because .vhdr is missing: {vhdr}")
            continue

        raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        duration_s = float(raw.times[-1])

        seg = raw.copy().crop(tmin=0.0, tmax=duration_s).load_data()
        n_pick = min(config.max_channels, len(seg.ch_names))
        seg.pick(list(range(n_pick)))

        data = seg.get_data()
        times = seg.times
        window = int(round(config.window_sec * sfreq))
        step = int(round(config.step_sec * sfreq))
        starts = list(range(0, data.shape[1] - window + 1, step))
        run_intervals = intervals[intervals["base"].astype(str) == base].copy()
        interval_list = [
            (float(r["onset_s"]), float(r["offset_s"])) for _, r in run_intervals.iterrows()
        ]

        for s0 in starts:
            s1 = s0 + window
            win = data[:, s0:s1]
            t_mid = float(times[s0:s1].mean())
            rms = float(np.sqrt(np.nanmean(win**2)))
            ptp = float(np.nanmax(win) - np.nanmin(win))
            ll = float(np.nanmean([_line_length(win[ch]) for ch in range(win.shape[0])]))

            y = int(any(a <= t_mid <= b for a, b in interval_list))
            feature_rows.append(
                {
                    "base": base,
                    "t_mid_s": t_mid,
                    "rms": rms,
                    "ptp": ptp,
                    "line_length": ll,
                    "y": y,
                }
            )

        info_rows.append(
            {
                "base": base,
                "sfreq": sfreq,
                "duration_s": duration_s,
                "window_sec": config.window_sec,
                "step_sec": config.step_sec,
                "n_channels_available": int(len(raw.ch_names)),
                "n_channels_used": int(n_pick),
                "n_windows": int(len(starts)),
                "n_positive_windows": int(
                    sum(1 for r in feature_rows if r["base"] == base and r["y"] == 1)
                ),
            }
        )

    features_df = pd.DataFrame(feature_rows)
    info_df = pd.DataFrame(info_rows)

    features_path = paths.outputs_dir / output_name
    info_path = paths.outputs_dir / output_info_name
    features_df.to_csv(features_path, index=False)
    info_df.to_csv(info_path, index=False)
    return features_df, info_df
