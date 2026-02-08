import re
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DATASET_ROOT = Path('EEG') / 'ds003029'
RUN_SUMMARY = Path('eda_outputs') / 'ds003029_run_summary.csv'
OUT_DIR = Path('eda_outputs')

# Same regex family used in eda_ds003029.ipynb
# NOTE: Keep onset/offset patterns mutually exclusive where possible.
# Previously `eeg\s*sz` caused strings like "eeg sz end" to be misclassified as onset.
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


def load_events_paths() -> pd.DataFrame:
    """Return DataFrame with columns: base, events_tsv (path string)."""
    if RUN_SUMMARY.exists():
        rs = pd.read_csv(RUN_SUMMARY)
        if 'events_tsv' in rs.columns:
            out = rs[['base', 'events_tsv']].copy()
            out['events_tsv'] = out['events_tsv'].fillna('')
            # Keep only non-empty
            out = out[out['events_tsv'].astype(str).str.len() > 0]
            return out

    # Fallback: discover all events.tsv under dataset
    paths = sorted(DATASET_ROOT.rglob('*_events.tsv'))
    rows = []
    for p in paths:
        base = p.with_name(p.name.replace('_events.tsv', '_ieeg'))
        rows.append({'base': str(base), 'events_tsv': str(p)})
    return pd.DataFrame(rows)


def read_events(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path, sep='\t')
    except Exception:
        return None


def normalize_path(p: str) -> Path:
    path = Path(p)
    if path.exists():
        return path
    # Many paths in run_summary are workspace-relative already
    alt = Path(p)
    if alt.exists():
        return alt
    # Try interpreting as relative to workspace root
    alt2 = Path.cwd() / p
    if alt2.exists():
        return alt2
    return path


def extract_intervals(events_df: pd.DataFrame) -> dict:
    """Extract onset/offset candidates and pair into intervals.

    Returns:
      - onsets: sorted list[float]
      - offsets: sorted list[float]
      - intervals: list[(onset, offset)] paired by earliest offset after each onset
      - intervals_typed: list[(onset, onset_type, offset, offset_type)]
      - matched_onset_types / matched_offset_types: Counter of trial_type strings matched
    """
    if events_df is None or 'trial_type' not in events_df.columns or 'onset' not in events_df.columns:
        return {
            'onsets': [],
            'offsets': [],
            'intervals': [],
            'intervals_typed': [],
            'matched_onset_types': Counter(),
            'matched_offset_types': Counter(),
            'trial_type_counts': Counter(),
        }

    tt = events_df['trial_type'].astype(str)
    onset_mask = tt.apply(lambda x: bool(ONSETS_RE.search(x))) & ~tt.apply(lambda x: bool(OFFSETS_RE.search(x)))
    offset_mask = tt.apply(lambda x: bool(OFFSETS_RE.search(x)))

    onset_types = tt[onset_mask]
    offset_types = tt[offset_mask]

    onsets_s = pd.to_numeric(events_df.loc[onset_mask, 'onset'], errors='coerce').astype(float)
    offsets_s = pd.to_numeric(events_df.loc[offset_mask, 'onset'], errors='coerce').astype(float)

    onset_events = (
        pd.DataFrame({'t': onsets_s, 'trial_type': onset_types})
        .dropna(subset=['t'])
        .sort_values('t', kind='mergesort')
        .reset_index(drop=True)
    )
    offset_events = (
        pd.DataFrame({'t': offsets_s, 'trial_type': offset_types})
        .dropna(subset=['t'])
        .sort_values('t', kind='mergesort')
        .reset_index(drop=True)
    )

    onsets = onset_events['t'].astype(float).tolist()
    offsets = offset_events['t'].astype(float).tolist()

    # Pairing: for each onset, take first offset strictly after onset
    intervals = []
    intervals_typed = []
    offset_idx = 0
    for onset_idx, onset in enumerate(onsets):
        while offset_idx < len(offsets) and offsets[offset_idx] <= onset:
            offset_idx += 1
        if offset_idx < len(offsets):
            offset = offsets[offset_idx]
            intervals.append((onset, offset))
            onset_type = str(onset_events.loc[onset_idx, 'trial_type'])
            offset_type = str(offset_events.loc[offset_idx, 'trial_type'])
            intervals_typed.append((onset, onset_type, offset, offset_type))
            offset_idx += 1

    # Trial type vocabulary counts
    trial_type_counts = Counter(tt.tolist())

    return {
        'onsets': onsets,
        'offsets': offsets,
        'intervals': intervals,
        'intervals_typed': intervals_typed,
        'matched_onset_types': Counter(onset_types.tolist()),
        'matched_offset_types': Counter(offset_types.tolist()),
        'trial_type_counts': trial_type_counts,
    }


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)

    df_paths = load_events_paths()
    if len(df_paths) == 0:
        print('No events.tsv paths found.')
        return 0

    per_run_rows = []
    interval_rows = []
    onset_vocab = Counter()
    offset_vocab = Counter()
    trial_vocab = Counter()

    for _, r in df_paths.iterrows():
        base = str(r.get('base', ''))
        evs = str(r.get('events_tsv', ''))
        evp = normalize_path(evs)
        if not evp.exists():
            per_run_rows.append(
                {
                    'base': base,
                    'events_tsv': evs,
                    'events_exists': False,
                    'n_trial_rows': np.nan,
                    'n_unique_trial_type': np.nan,
                    'n_onset_markers': np.nan,
                    'n_offset_markers': np.nan,
                    'n_intervals_paired': np.nan,
                    'multi_seizure_candidate': np.nan,
                    'has_unpaired_onset': np.nan,
                    'has_orphan_offset': np.nan,
                    'min_onset_s': np.nan,
                    'max_offset_s': np.nan,
                }
            )
            continue

        evdf = read_events(evp)
        info = extract_intervals(evdf)

        onsets = info['onsets']
        offsets = info['offsets']
        intervals = info['intervals']
        intervals_typed = info['intervals_typed']

        onset_vocab.update(info['matched_onset_types'])
        offset_vocab.update(info['matched_offset_types'])
        trial_vocab.update(info['trial_type_counts'])

        n_on = len(onsets)
        n_off = len(offsets)
        n_int = len(intervals)

        # Heuristics / flags
        multi = (n_on >= 2) or (n_off >= 2) or (n_int >= 2)
        has_unpaired_onset = n_on > n_int
        # Orphan offset: offsets not used for pairing (including those before first onset)
        used_offsets = set([b for _, b in intervals])
        orphan_offset = any([(o not in used_offsets) for o in offsets])

        per_run_rows.append(
            {
                'base': base,
                'events_tsv': str(evp),
                'events_exists': True,
                'n_trial_rows': int(len(evdf)) if evdf is not None else 0,
                'n_unique_trial_type': int(evdf['trial_type'].astype(str).nunique())
                if evdf is not None and 'trial_type' in evdf.columns
                else 0,
                'n_onset_markers': n_on,
                'n_offset_markers': n_off,
                'n_intervals_paired': n_int,
                'multi_seizure_candidate': bool(multi),
                'has_unpaired_onset': bool(has_unpaired_onset),
                'has_orphan_offset': bool(orphan_offset),
                'min_onset_s': float(onsets[0]) if n_on else np.nan,
                'max_offset_s': float(max([b for _, b in intervals])) if n_int else np.nan,
            }
        )

        for (onset, onset_type, offset, offset_type) in intervals_typed:
            interval_rows.append(
                {
                    'base': base,
                    'events_tsv': str(evp),
                    'onset_s': float(onset),
                    'offset_s': float(offset),
                    'duration_s': float(offset - onset),
                    'onset_trial_type': onset_type,
                    'offset_trial_type': offset_type,
                }
            )

    per_run = pd.DataFrame(per_run_rows)

    # Summary prints
    print('Runs with events.tsv listed:', len(df_paths))
    print('Runs with events.tsv readable:', int(per_run['events_exists'].sum()))

    # Distribution stats
    ok = per_run[per_run['events_exists'] == True]
    if len(ok) > 0:
        print('Runs with >=1 onset markers:', int((ok['n_onset_markers'] > 0).sum()))
        print('Runs with >=1 offset markers:', int((ok['n_offset_markers'] > 0).sum()))
        print('Runs with paired intervals >=1:', int((ok['n_intervals_paired'] > 0).sum()))
        print('Multi-seizure candidates:', int(ok['multi_seizure_candidate'].sum()))
        print('Has unpaired onset:', int(ok['has_unpaired_onset'].sum()))
        print('Has orphan offset:', int(ok['has_orphan_offset'].sum()))

    out_by_run = OUT_DIR / 'ds003029_marker_qc_by_run.csv'
    per_run.to_csv(out_by_run, index=False)
    print('Wrote:', out_by_run.resolve())

    out_intervals = OUT_DIR / 'ds003029_seizure_intervals_by_run.csv'
    pd.DataFrame(interval_rows).to_csv(out_intervals, index=False)
    print('Wrote:', out_intervals.resolve())

    # Vocab exports (top N)
    def counter_to_df(c: Counter, topn: int = 200) -> pd.DataFrame:
        items = c.most_common(topn)
        return pd.DataFrame(items, columns=['trial_type', 'count'])

    out_on = OUT_DIR / 'ds003029_trial_type_onset_vocab.csv'
    out_off = OUT_DIR / 'ds003029_trial_type_offset_vocab.csv'
    out_all = OUT_DIR / 'ds003029_trial_type_vocab.csv'

    counter_to_df(onset_vocab).to_csv(out_on, index=False)
    counter_to_df(offset_vocab).to_csv(out_off, index=False)
    counter_to_df(trial_vocab).to_csv(out_all, index=False)

    print('Wrote:', out_on.resolve())
    print('Wrote:', out_off.resolve())
    print('Wrote:', out_all.resolve())

    # Convenience: show the most common matched strings
    print('\nTop matched onset trial_type strings:')
    for s, c in onset_vocab.most_common(15):
        print(f'  {c:5d}  {s}')

    print('\nTop matched offset trial_type strings:')
    for s, c in offset_vocab.most_common(15):
        print(f'  {c:5d}  {s}')

    print('\nNext step suggestion:')
    print('- Use ds003029_marker_qc_by_run.csv to identify runs with multi_seizure_candidate=1 or pairing issues.')
    print('- For window-level ground truth, consider labeling windows as ictal if they overlap ANY interval among the paired intervals in a run (union of intervals).')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
