from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .markers import parse_events_df, pair_intervals


@dataclass(frozen=True)
class MarkerQcOutputs:
    per_run: pd.DataFrame
    intervals: pd.DataFrame
    onset_vocab: pd.DataFrame
    offset_vocab: pd.DataFrame
    trial_vocab: pd.DataFrame


def read_events_tsv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return None


def build_marker_qc(events_paths: pd.DataFrame) -> MarkerQcOutputs:
    """Build marker QC tables from a DF with columns: base, events_tsv."""

    per_run_rows: list[dict] = []
    interval_rows: list[dict] = []

    onset_vocab = Counter()
    offset_vocab = Counter()
    trial_vocab = Counter()

    for _, r in events_paths.iterrows():
        base = str(r.get("base", ""))
        events_tsv = str(r.get("events_tsv", ""))
        evp = Path(events_tsv)

        if not evp.exists():
            per_run_rows.append(
                {
                    "base": base,
                    "events_tsv": events_tsv,
                    "events_exists": False,
                    "n_trial_rows": np.nan,
                    "n_unique_trial_type": np.nan,
                    "n_onset_markers": np.nan,
                    "n_offset_markers": np.nan,
                    "n_intervals_paired": np.nan,
                    "multi_seizure_candidate": np.nan,
                    "has_unpaired_onset": np.nan,
                    "has_orphan_offset": np.nan,
                    "min_onset_s": np.nan,
                    "max_offset_s": np.nan,
                }
            )
            continue

        evdf = read_events_tsv(evp)
        parsed = parse_events_df(evdf)
        on = parsed.onset_events
        off = parsed.offset_events

        onset_vocab.update(on["trial_type"].astype(str).tolist())
        offset_vocab.update(off["trial_type"].astype(str).tolist())
        trial_vocab.update(parsed.trial_vocab)

        intervals = pair_intervals(on, off)
        n_on = int(len(on))
        n_off = int(len(off))
        n_int = int(len(intervals))

        multi = (n_on >= 2) or (n_off >= 2) or (n_int >= 2)
        has_unpaired_onset = n_on > n_int
        used_offsets = set([b for _, b in intervals])
        orphan_offset = any([(float(o) not in used_offsets) for o in off["t"].astype(float).tolist()])

        per_run_rows.append(
            {
                "base": base,
                "events_tsv": str(evp),
                "events_exists": True,
                "n_trial_rows": int(len(evdf)) if evdf is not None else 0,
                "n_unique_trial_type": int(evdf["trial_type"].astype(str).nunique())
                if evdf is not None and "trial_type" in evdf.columns
                else 0,
                "n_onset_markers": n_on,
                "n_offset_markers": n_off,
                "n_intervals_paired": n_int,
                "multi_seizure_candidate": bool(multi),
                "has_unpaired_onset": bool(has_unpaired_onset),
                "has_orphan_offset": bool(orphan_offset),
                "min_onset_s": float(on["t"].min()) if n_on else np.nan,
                "max_offset_s": float(max([b for _, b in intervals])) if n_int else np.nan,
            }
        )

        if len(intervals) > 0:
            # Recover typed strings for paired intervals by position
            # (safe because pair_intervals uses onset order and first offset-after-onset)
            offset_events = off.reset_index(drop=True)
            offset_ptr = 0
            for onset_idx, (a, b) in enumerate(intervals):
                while offset_ptr < len(offset_events) and float(offset_events.loc[offset_ptr, "t"]) <= float(a):
                    offset_ptr += 1
                if offset_ptr >= len(offset_events):
                    break
                interval_rows.append(
                    {
                        "base": base,
                        "events_tsv": str(evp),
                        "onset_s": float(a),
                        "offset_s": float(b),
                        "duration_s": float(b - a),
                        "onset_trial_type": str(on.loc[onset_idx, "trial_type"]) if onset_idx < len(on) else "",
                        "offset_trial_type": str(offset_events.loc[offset_ptr, "trial_type"]),
                    }
                )
                offset_ptr += 1

    per_run = pd.DataFrame(per_run_rows)
    intervals_df = pd.DataFrame(interval_rows)

    def counter_to_df(c: Counter, topn: int = 300) -> pd.DataFrame:
        items = c.most_common(topn)
        return pd.DataFrame(items, columns=["trial_type", "count"])

    return MarkerQcOutputs(
        per_run=per_run,
        intervals=intervals_df,
        onset_vocab=counter_to_df(onset_vocab),
        offset_vocab=counter_to_df(offset_vocab),
        trial_vocab=counter_to_df(trial_vocab),
    )


def export_marker_qc(out_dir: Path, outputs: MarkerQcOutputs) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "ds003029_marker_qc_by_run.csv").write_text(outputs.per_run.to_csv(index=False), encoding="utf-8")
    (out_dir / "ds003029_seizure_intervals_by_run.csv").write_text(outputs.intervals.to_csv(index=False), encoding="utf-8")
    (out_dir / "ds003029_trial_type_onset_vocab.csv").write_text(outputs.onset_vocab.to_csv(index=False), encoding="utf-8")
    (out_dir / "ds003029_trial_type_offset_vocab.csv").write_text(outputs.offset_vocab.to_csv(index=False), encoding="utf-8")
    (out_dir / "ds003029_trial_type_vocab.csv").write_text(outputs.trial_vocab.to_csv(index=False), encoding="utf-8")
