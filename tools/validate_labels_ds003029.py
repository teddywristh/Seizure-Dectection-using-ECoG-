import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Make `src/` importable when running as a script
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ds003029_eda.data.io import resolve_artifact_path  # noqa: E402
from ds003029_eda.markers import first_onset_offset  # noqa: E402
from ds003029_eda.paths import get_paths, resolve_workspace_root  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Validate seizure labels in ds003029 against readable events.tsv markers.')
    parser.add_argument('--workspace-root', default=None)
    return parser


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
    args = build_parser().parse_args()
    paths = get_paths(resolve_workspace_root(args.workspace_root))
    run_summary = paths.outputs_dir / 'ds003029_run_summary.csv'
    if not run_summary.exists():
        raise FileNotFoundError(f"Missing {run_summary.resolve()}")

    rs = pd.read_csv(run_summary)
    if 'events_tsv' not in rs.columns:
        raise KeyError("run_summary missing 'events_tsv' column")

    rs['events_tsv'] = rs['events_tsv'].fillna('')

    rows = []  # type: list[dict]
    for _, r in rs.iterrows():
        ev = r.get('events_tsv', '')
        if not isinstance(ev, str) or ev.strip() == '':
            continue

        evp = resolve_artifact_path(ev, workspace=paths.workspace, outputs_dir=paths.outputs_dir)
        if not evp.exists():
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
