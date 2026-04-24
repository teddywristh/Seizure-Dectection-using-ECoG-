from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from ..paths import WorkspacePaths, get_paths


SIGNAL_CHANNEL_TYPES = {"ECOG", "SEEG", "EEG", "IEEG"}
NON_SIGNAL_MNE_TYPES = {"misc", "stim", "ecg", "eog", "emg", "resp", "bio"}


def _ensure_mne_workspace_home(paths: WorkspacePaths) -> Path:
    mne_home = paths.workspace / ".mne"
    mne_home.mkdir(parents=True, exist_ok=True)
    os.environ["_MNE_FAKE_HOME_DIR"] = str(paths.workspace)
    os.environ["MNE_HOME"] = str(mne_home)
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    return mne_home


def load_mne(paths: WorkspacePaths):
    _ensure_mne_workspace_home(paths)
    import mne

    return mne


def resolve_existing_path(path_str: str, workspace: Path | None = None) -> Path:
    path = Path(path_str)
    if path.exists():
        return path

    if workspace is not None:
        alt = workspace / path
        if alt.exists():
            return alt

    alt = Path.cwd() / path
    if alt.exists():
        return alt
    return path


def read_run_inventory(
    paths: WorkspacePaths | None = None,
    *,
    require_intervals: bool = True,
    require_eeg_content: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = paths or get_paths()

    run_summary_path = paths.outputs_dir / "ds003029_run_summary.csv"
    intervals_path = paths.outputs_dir / "ds003029_seizure_intervals_by_run.csv"
    if not run_summary_path.exists():
        raise FileNotFoundError(f"Missing run summary: {run_summary_path}")
    if not intervals_path.exists():
        raise FileNotFoundError(f"Missing interval summary: {intervals_path}")

    runs = pd.read_csv(run_summary_path)
    intervals = pd.read_csv(intervals_path)

    if require_eeg_content and "eeg_content_present" in runs.columns:
        runs = runs[runs["eeg_content_present"].astype(bool)].copy()

    if require_intervals:
        interval_bases = set(intervals["base"].astype(str))
        runs = runs[runs["base"].astype(str).isin(interval_bases)].copy()

    interval_counts = intervals.groupby("base").size().rename("n_intervals")
    runs["n_intervals"] = runs["base"].map(interval_counts).fillna(0).astype(int)
    runs = runs.sort_values(["subject", "session", "run", "base"], kind="mergesort").reset_index(drop=True)
    return runs, intervals


def read_channels_tsv(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return None


def read_ieeg_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _select_signal_channels(raw, channels_df: pd.DataFrame | None) -> tuple[list[str], list[str]]:
    if channels_df is not None and {"name", "type"}.issubset(channels_df.columns):
        names = channels_df["name"].astype(str)
        types = channels_df["type"].astype(str).str.upper()
        signal_names = [
            name for name, ch_type in zip(names.tolist(), types.tolist()) if ch_type in SIGNAL_CHANNEL_TYPES and name in raw.ch_names
        ]

        declared_bads: list[str] = []
        if "status" in channels_df.columns:
            status = channels_df["status"].astype(str).str.lower()
            declared_bads = [
                name for name, is_bad in zip(names.tolist(), status.eq("bad").tolist()) if is_bad and name in raw.ch_names
            ]

        declared_bads = [name for name in declared_bads if name in signal_names]

        if signal_names:
            return signal_names, declared_bads

    channel_types = raw.get_channel_types()
    signal_names = [
        name
        for name, channel_type in zip(raw.ch_names, channel_types)
        if str(channel_type).lower() not in NON_SIGNAL_MNE_TYPES
    ]
    return signal_names, []


def load_raw_run(
    row: pd.Series | dict[str, Any],
    *,
    paths: WorkspacePaths | None = None,
    preload: bool = True,
):
    paths = paths or get_paths()
    mne = load_mne(paths)

    base = str(row.get("base", ""))
    vhdr_path = resolve_existing_path(str(row.get("vhdr") or f"{base}.vhdr"), workspace=paths.workspace)
    if not vhdr_path.exists():
        raise FileNotFoundError(f"Missing BrainVision header for run {base}: {vhdr_path}")

    channels_path_str = str(row.get("channels_tsv") or "")
    channels_path = resolve_existing_path(channels_path_str, workspace=paths.workspace) if channels_path_str else None
    ieeg_json_path_str = str(row.get("ieeg_json") or "")
    ieeg_json_path = resolve_existing_path(ieeg_json_path_str, workspace=paths.workspace) if ieeg_json_path_str else None

    raw = mne.io.read_raw_brainvision(vhdr_path, preload=preload, verbose="ERROR")
    channels_df = read_channels_tsv(channels_path)
    signal_names, declared_bads = _select_signal_channels(raw, channels_df)
    if signal_names:
        raw.pick(signal_names)

    valid_channel_names = set(raw.ch_names)
    raw.info["bads"] = sorted(
        channel
        for channel in set(raw.info.get("bads", [])).union(set(declared_bads))
        if channel in valid_channel_names
    )
    meta = {
        "base": base,
        "vhdr_path": str(vhdr_path),
        "channels_tsv_path": str(channels_path) if channels_path is not None else "",
        "ieeg_json_path": str(ieeg_json_path) if ieeg_json_path is not None else "",
        "ieeg_json": read_ieeg_json(ieeg_json_path),
        "declared_bads": sorted(set(declared_bads)),
        "n_channels_loaded": int(len(raw.ch_names)),
    }
    return raw, meta


def intervals_for_base(intervals_df: pd.DataFrame, base: str) -> list[tuple[float, float]]:
    subset = intervals_df[intervals_df["base"].astype(str) == str(base)].copy()
    if subset.empty:
        return []
    return [
        (float(row["onset_s"]), float(row["offset_s"]))
        for _, row in subset.sort_values("onset_s", kind="mergesort").iterrows()
        if pd.notna(row.get("onset_s")) and pd.notna(row.get("offset_s"))
    ]


def line_frequency_from_row(row: pd.Series | dict[str, Any], ieeg_json: dict[str, Any] | None = None) -> float | None:
    for candidate in [row.get("line_freq"), row.get("PowerLineFrequency")]:
        if candidate is None or candidate == "":
            continue
        try:
            value = float(candidate)
        except Exception:
            continue
        if value > 0:
            return value

    meta = ieeg_json or {}
    for key in ["PowerLineFrequency", "line_frequency", "LineFrequency"]:
        if key in meta:
            try:
                value = float(meta[key])
            except Exception:
                continue
            if value > 0:
                return value
    return None