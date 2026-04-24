from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from ds003029_eda.marker_qc import build_marker_qc, export_marker_qc
from ds003029_eda.paths import get_paths


def _has_real_eeg_data(workspace_root: Path) -> bool:
    eeg_root = workspace_root / "EEG" / "ds003029"
    if not eeg_root.exists():
        return False
    try:
        next(eeg_root.rglob("*.eeg"))
    except StopIteration:
        return False
    return True


def _resolve_workspace_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    for candidate in [repo_root.parent, repo_root]:
        if _has_real_eeg_data(candidate):
            return candidate.resolve()
    for candidate in [repo_root.parent, repo_root]:
        if (candidate / "EEG" / "ds003029").exists():
            return candidate.resolve()
    return repo_root.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build ds003029 marker QC tables for the selected workspace root.")
    parser.add_argument("--workspace-root", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workspace_root = _resolve_workspace_root(args.workspace_root)
    paths = get_paths(workspace_root)
    run_summary_path = paths.outputs_dir / "ds003029_run_summary.csv"
    if not run_summary_path.exists():
        raise FileNotFoundError(f"Missing run summary: {run_summary_path}")

    run_summary = pd.read_csv(run_summary_path)
    if not {"base", "events_tsv"}.issubset(run_summary.columns):
        raise KeyError("run_summary must contain base and events_tsv columns")

    outputs = build_marker_qc(run_summary[["base", "events_tsv"]].copy())
    export_marker_qc(paths.outputs_dir, outputs)
    print(f"Workspace root: {workspace_root.as_posix()}")
    print(f"Marker QC exported: {len(outputs.per_run)} runs")
    print(f"Paired seizure intervals: {len(outputs.intervals)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())