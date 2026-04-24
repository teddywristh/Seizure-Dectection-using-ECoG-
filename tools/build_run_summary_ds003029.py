from __future__ import annotations

import argparse
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from ds003029_eda.paths import get_paths
from ds003029_eda.run_summary import build_run_summary, export_run_summary


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
    parser = argparse.ArgumentParser(description="Build ds003029 run summary for the selected workspace root.")
    parser.add_argument("--workspace-root", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workspace_root = _resolve_workspace_root(args.workspace_root)
    paths = get_paths(workspace_root)
    result = build_run_summary(paths.dataset_root)
    export_run_summary(paths.outputs_dir, result)
    print(f"Workspace root: {workspace_root.as_posix()}")
    print(f"Run summary exported: {len(result.run_summary)} runs")
    print(f"Event vocabulary entries: {len(result.event_vocab)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())