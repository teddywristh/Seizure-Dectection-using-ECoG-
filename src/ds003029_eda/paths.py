from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkspacePaths:
    workspace: Path
    dataset_root: Path
    outputs_dir: Path


def has_real_eeg_data(workspace_root: Path) -> bool:
    eeg_root = workspace_root / "EEG" / "ds003029"
    if not eeg_root.exists():
        return False
    try:
        next(eeg_root.rglob("*.eeg"))
    except StopIteration:
        return False
    return True


def find_workspace_root(start: Path | None = None) -> Path:
    """Find the repo/workspace root by searching for EEG/ds003029."""
    start = (start or Path.cwd()).resolve()
    for p in [start, *start.parents]:
        if (p / "EEG" / "ds003029").exists():
            return p
    return start


def resolve_workspace_root(workspace: Path | str | None = None) -> Path:
    if workspace is not None:
        return Path(workspace).resolve()

    start = Path.cwd().resolve()
    candidates = [start, *start.parents]
    for candidate in candidates:
        if has_real_eeg_data(candidate):
            return candidate
    for candidate in candidates:
        if (candidate / "EEG" / "ds003029").exists():
            return candidate
    return find_workspace_root(start)


def get_paths(workspace: Path | None = None) -> WorkspacePaths:
    ws = resolve_workspace_root(workspace)
    return WorkspacePaths(
        workspace=ws,
        dataset_root=ws / "EEG" / "ds003029",
        outputs_dir=ws / "eda_outputs",
    )
