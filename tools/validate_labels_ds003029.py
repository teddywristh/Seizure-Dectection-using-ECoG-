from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Make `src/` importable when running as a script
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from ds003029_eda.markers import first_onset_offset  # noqa: E402


# Workspace-relative paths
DATASET_ROOT = Path('EEG') / 'ds003029'
RUN_SUMMARY = Path('eda_outputs') / 'ds003029_run_summary.csv'


def parse_events_tsv(path: Path) -> Optional[Dict]:
    try:
        df = pd.read_csv(path, sep='\t')
    except Exception:
        return None

    if 'trial_type' not in df.columns or 'onset' not in df.columns:
        return None

    onset, offset = first_onset_offset(df)
    return {
        'events_has_onset': onset is not None,
        'events_has_offset': offset is not None,
        'events_onset': float(onset) if onset is not None else np.nan,
        'events_offset': float(offset) if offset is not None else np.nan,
    }


def main() -> int:
    if not RUN_SUMMARY.exists():
        raise FileNotFoundError(f"Missing {RUN_SUMMARY.resolve()}")

    rs = pd.read_csv(RUN_SUMMARY)
    if 'events_tsv' not in rs.columns:
        raise KeyError("run_summary missing 'events_tsv' column")

    rs['events_tsv'] = rs['events_tsv'].fillna('')

    rows = []  # type: list[dict]
    for _, r in rs.iterrows():
        ev = r.get('events_tsv', '')
        if not isinstance(ev, str) or ev.strip() == '':
            continue

        evp = Path(ev)
        if not evp.is_absolute():
            # In run_summary, events_tsv is usually already workspace-relative like 'EEG\\ds003029\\...'
            # If it is relative, resolve it under workspace.
            evp = Path(evp)
        # Ensure existence
        if not evp.exists():
            # Try resolving from workspace root (cwd)
            evp2 = Path(ev)
            if evp2.exists():
                evp = evp2
            else:
                continue

        parsed = parse_events_tsv(evp)
        if parsed is None:
            continue

        rows.append(
            {
                'base': r.get('base', ''),
                'events_tsv': str(evp),
                'summary_onset': r.get('seizure_onset_s', np.nan),
                'summary_offset': r.get('seizure_offset_s', np.nan),
                **parsed,
            }
        )

    cmp = pd.DataFrame(rows)
    print('Runs in run_summary:', len(rs))
    print('Runs with readable events.tsv:', len(cmp))
    if len(cmp) == 0:
        return 0

    cmp['onset_diff_s'] = cmp['summary_onset'] - cmp['events_onset']
    cmp['offset_diff_s'] = cmp['summary_offset'] - cmp['events_offset']

    tol = 0.5
    onset_mask = np.isfinite(cmp['summary_onset']) & np.isfinite(cmp['events_onset'])
    offset_mask = np.isfinite(cmp['summary_offset']) & np.isfinite(cmp['events_offset'])

    onset_match = (cmp.loc[onset_mask, 'onset_diff_s'].abs() <= tol)
    offset_match = (cmp.loc[offset_mask, 'offset_diff_s'].abs() <= tol)

    onset_rate = float(onset_match.mean() * 100) if len(onset_match) else float('nan')
    offset_rate = float(offset_match.mean() * 100) if len(offset_match) else float('nan')

    print(f'Onset comparable: {int(onset_mask.sum())} | match@{tol}s: {int(onset_match.sum())} ({onset_rate:.1f}%)')
    print(f'Offset comparable: {int(offset_mask.sum())} | match@{tol}s: {int(offset_match.sum())} ({offset_rate:.1f}%)')

    bad_order = cmp[
        np.isfinite(cmp['summary_onset'])
        & np.isfinite(cmp['summary_offset'])
        & (cmp['summary_offset'] <= cmp['summary_onset'])
    ]
    print('Runs with summary_offset <= summary_onset:', len(bad_order))

    if onset_mask.sum() > 0:
        worst = (
            cmp.loc[onset_mask]
            .assign(absdiff=lambda d: d['onset_diff_s'].abs())
            .sort_values('absdiff', ascending=False)
            .head(10)
        )
        print('\nTop onset mismatches (seconds):')
        print(worst[['base', 'summary_onset', 'events_onset', 'onset_diff_s']].to_string(index=False))

    if offset_mask.sum() > 0:
        worst = (
            cmp.loc[offset_mask]
            .assign(absdiff=lambda d: d['offset_diff_s'].abs())
            .sort_values('absdiff', ascending=False)
            .head(10)
        )
        print('\nTop offset mismatches (seconds):')
        print(worst[['base', 'summary_offset', 'events_offset', 'offset_diff_s']].to_string(index=False))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
