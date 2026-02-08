from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkspacePaths:
    workspace: Path
    dataset_root: Path
    outputs_dir: Path


def find_workspace_root(start: Path | None = None) -> Path:
    """Find the repo/workspace root by searching for EEG/ds003029."""
    start = (start or Path.cwd()).resolve()
    for p in [start, *start.parents]:
        if (p / "EEG" / "ds003029").exists():
            return p
    return start


def get_paths(workspace: Path | None = None) -> WorkspacePaths:
    ws = (workspace or find_workspace_root()).resolve()
    return WorkspacePaths(
        workspace=ws,
        dataset_root=ws / "EEG" / "ds003029",
        outputs_dir=ws / "eda_outputs",
    )
