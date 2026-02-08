from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

# NOTE: Keep onset/offset patterns mutually exclusive where possible.
# In ds003029, strings like "eeg sz end" must NOT be misclassified as onset.
ONSETS_RE = re.compile(
    r"(?i)(?:"
    r"\bonset\b|"
    r"sz\s*onset|"
    r"seizure\s*onset|"
    r"\bsz\s*event\b|"
    r"eeg\s*sz\s*(?:start|onset)|"
    r"electrographic\s*(?:onset|ons\b)"
    r")"
)
OFFSETS_RE = re.compile(
    r"(?i)(?:"
    r"\boffset\b|"
    r"sz\s*offset|"
    r"seizure\s*offset|"
    r"electrographic\s*end|"
    r"eeg\s*sz\s*end"
    r")"
)


def is_onset_trial_type(trial_type: str) -> bool:
    s = str(trial_type)
    return bool(ONSETS_RE.search(s)) and not bool(OFFSETS_RE.search(s))


def is_offset_trial_type(trial_type: str) -> bool:
    return bool(OFFSETS_RE.search(str(trial_type)))


@dataclass(frozen=True)
class ParsedEvents:
    onset_events: pd.DataFrame  # columns: t, trial_type
    offset_events: pd.DataFrame  # columns: t, trial_type
    trial_vocab: Counter


def parse_events_df(events_df: pd.DataFrame) -> ParsedEvents:
    if events_df is None or len(events_df) == 0:
        return ParsedEvents(
            onset_events=pd.DataFrame(columns=["t", "trial_type"]),
            offset_events=pd.DataFrame(columns=["t", "trial_type"]),
            trial_vocab=Counter(),
        )

    if "trial_type" not in events_df.columns or "onset" not in events_df.columns:
        return ParsedEvents(
            onset_events=pd.DataFrame(columns=["t", "trial_type"]),
            offset_events=pd.DataFrame(columns=["t", "trial_type"]),
            trial_vocab=Counter(events_df.columns.tolist()),
        )

    tt = events_df["trial_type"].astype(str)
    onset_mask = tt.apply(is_onset_trial_type)
    offset_mask = tt.apply(is_offset_trial_type)

    onsets_s = pd.to_numeric(events_df.loc[onset_mask, "onset"], errors="coerce").astype(float)
    offsets_s = pd.to_numeric(events_df.loc[offset_mask, "onset"], errors="coerce").astype(float)

    onset_events = (
        pd.DataFrame({"t": onsets_s, "trial_type": tt[onset_mask]})
        .dropna(subset=["t"])
        .sort_values("t", kind="mergesort")
        .reset_index(drop=True)
    )
    offset_events = (
        pd.DataFrame({"t": offsets_s, "trial_type": tt[offset_mask]})
        .dropna(subset=["t"])
        .sort_values("t", kind="mergesort")
        .reset_index(drop=True)
    )

    return ParsedEvents(
        onset_events=onset_events,
        offset_events=offset_events,
        trial_vocab=Counter(tt.tolist()),
    )


def pair_intervals(onset_events: pd.DataFrame, offset_events: pd.DataFrame) -> list[tuple[float, float]]:
    """Pair each onset to the first offset strictly after it."""
    onsets = onset_events["t"].astype(float).tolist() if len(onset_events) else []
    offsets = offset_events["t"].astype(float).tolist() if len(offset_events) else []

    intervals: list[tuple[float, float]] = []
    offset_idx = 0
    for onset in onsets:
        while offset_idx < len(offsets) and offsets[offset_idx] <= onset:
            offset_idx += 1
        if offset_idx < len(offsets):
            intervals.append((float(onset), float(offsets[offset_idx])))
            offset_idx += 1
    return intervals


def first_onset_offset(events_df: pd.DataFrame) -> tuple[float | None, float | None]:
    parsed = parse_events_df(events_df)
    on = parsed.onset_events
    off = parsed.offset_events
    if len(on) == 0:
        return None, None

    onset = float(on.loc[0, "t"])
    offset = None
    if len(off) > 0:
        later = off[off["t"] > onset]
        if len(later) > 0:
            offset = float(later.iloc[0]["t"])
    return onset, offset


def trial_type_vocab(dfs: Iterable[pd.DataFrame]) -> pd.DataFrame:
    vocab = Counter()
    for df in dfs:
        if df is None or "trial_type" not in df.columns:
            continue
        vocab.update(df["trial_type"].astype(str).tolist())
    out = pd.DataFrame(vocab.most_common(), columns=["trial_type", "count"])
    return out
