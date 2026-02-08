from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .markers import first_onset_offset


@dataclass(frozen=True)
class RunSummaryResult:
    run_summary: pd.DataFrame
    event_vocab: pd.DataFrame


def parse_bids_entities_from_filename(fname: str) -> dict[str, str]:
    """Parse a subset of BIDS entities from a filename.

    Supports names like:
      sub-jh101_ses-presurgery_task-ictal_acq-ecog_run-01_ieeg.vhdr
    """
    stem = Path(fname).name
    for ext in [".vhdr", ".vmrk", ".eeg", ".tsv", ".json"]:
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break

    parts = stem.split("_")
    out: dict[str, str] = {}
    for p in parts:
        if p.startswith("sub-"):
            out["subject"] = p.replace("sub-", "")
        elif p.startswith("ses-"):
            out["session"] = p.replace("ses-", "")
        elif p.startswith("task-"):
            out["task"] = p.replace("task-", "")
        elif p.startswith("acq-"):
            out["acq"] = p.replace("acq-", "")
        elif p.startswith("run-"):
            out["run"] = p.replace("run-", "")
    return out


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_tsv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return None


def _get_sampling_frequency(ieeg_json: dict) -> float | None:
    candidates = [
        "SamplingFrequency",
        "sampling_frequency",
        "sfreq",
        "SamplingRate",
    ]
    for k in candidates:
        if k in ieeg_json:
            try:
                v = float(ieeg_json[k])
                if np.isfinite(v) and v > 0:
                    return v
            except Exception:
                continue
    return None


def _get_line_frequency(ieeg_json: dict) -> float | None:
    candidates = ["PowerLineFrequency", "line_frequency", "LineFrequency"]
    for k in candidates:
        if k in ieeg_json:
            try:
                v = float(ieeg_json[k])
                if np.isfinite(v) and v > 0:
                    return v
            except Exception:
                continue
    return None


def build_run_summary(dataset_root: Path) -> RunSummaryResult:
    dataset_root = Path(dataset_root)

    vhdr_paths = sorted(dataset_root.glob("sub-*/ses-*/ieeg/*_ieeg.vhdr"))
    rows: list[dict] = []
    vocab = {}

    trial_type_counter: dict[str, int] = {}

    for vhdr in vhdr_paths:
        base = vhdr.with_suffix("")
        channels_tsv = vhdr.with_name(vhdr.name.replace("_ieeg.vhdr", "_channels.tsv"))
        events_tsv = vhdr.with_name(vhdr.name.replace("_ieeg.vhdr", "_events.tsv"))
        ieeg_json = vhdr.with_name(vhdr.name.replace("_ieeg.vhdr", "_ieeg.json"))
        eeg_bin = base.with_suffix(".eeg")
        vmrk = base.with_suffix(".vmrk")

        ent = parse_bids_entities_from_filename(vhdr.name)
        subject = ent.get("subject") or vhdr.parents[2].name.replace("sub-", "")
        session = ent.get("session") or vhdr.parents[1].name.replace("ses-", "")

        # Channels
        ch_df = _read_tsv(channels_tsv) if channels_tsv.exists() else None
        n_channels = int(len(ch_df)) if ch_df is not None else np.nan
        n_bad = np.nan
        if ch_df is not None and "status" in ch_df.columns:
            n_bad = int((ch_df["status"].astype(str).str.lower() == "bad").sum())

        # iEEG JSON
        meta = _read_json(ieeg_json) if ieeg_json.exists() else {}
        sfreq = _get_sampling_frequency(meta)
        line_freq = _get_line_frequency(meta)

        # Events
        ev_df = _read_tsv(events_tsv) if events_tsv.exists() else None
        onset, offset = (None, None)
        has_onset = False
        has_offset = False
        if ev_df is not None:
            onset, offset = first_onset_offset(ev_df)
            has_onset = onset is not None
            has_offset = offset is not None
            if "trial_type" in ev_df.columns:
                for s in ev_df["trial_type"].astype(str).tolist():
                    trial_type_counter[s] = trial_type_counter.get(s, 0) + 1

        seizure_duration_s = float(offset - onset) if (onset is not None and offset is not None) else np.nan

        eeg_size_bytes = np.nan
        eeg_content_present = False
        if eeg_bin.exists():
            try:
                eeg_size_bytes = float(eeg_bin.stat().st_size)
                eeg_content_present = bool(eeg_size_bytes > 1024 * 1024)  # heuristic: >1MB is likely real content
            except Exception:
                eeg_content_present = False

        rows.append(
            {
                "subject": subject,
                "session": session,
                "task": ent.get("task", ""),
                "acq": ent.get("acq", ""),
                "run": ent.get("run", ""),
                "base": str(base),
                "vhdr": str(vhdr),
                "vmrk": str(vmrk) if vmrk.exists() else "",
                "eeg": str(eeg_bin) if eeg_bin.exists() else "",
                "channels_tsv": str(channels_tsv) if channels_tsv.exists() else "",
                "events_tsv": str(events_tsv) if events_tsv.exists() else "",
                "ieeg_json": str(ieeg_json) if ieeg_json.exists() else "",
                "sfreq": sfreq if sfreq is not None else np.nan,
                "line_freq": line_freq if line_freq is not None else np.nan,
                "n_channels": n_channels,
                "n_bad": n_bad,
                "has_onset": bool(has_onset),
                "has_offset": bool(has_offset),
                "seizure_onset_s": float(onset) if onset is not None else np.nan,
                "seizure_offset_s": float(offset) if offset is not None else np.nan,
                "seizure_duration_s": seizure_duration_s,
                "eeg_size_bytes": eeg_size_bytes,
                "eeg_content_present": bool(eeg_content_present),
            }
        )

    run_summary = pd.DataFrame(rows)
    event_vocab = pd.DataFrame(sorted(trial_type_counter.items(), key=lambda kv: kv[1], reverse=True), columns=["trial_type", "count"])
    return RunSummaryResult(run_summary=run_summary, event_vocab=event_vocab)


def export_run_summary(out_dir: Path, result: RunSummaryResult) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ds003029_run_summary.csv").write_text(result.run_summary.to_csv(index=False), encoding="utf-8")
    (out_dir / "ds003029_event_vocab.csv").write_text(result.event_vocab.to_csv(index=False), encoding="utf-8")
