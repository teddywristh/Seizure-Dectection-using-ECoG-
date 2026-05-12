from __future__ import annotations

from pathlib import Path

import pytest

from ds003029_eda.experiments.dl import _resolve_raw_loading_strategy


def test_resolve_raw_loading_strategy_auto_falls_back_to_on_demand(tmp_path: Path) -> None:
    assert _resolve_raw_loading_strategy(strategy="auto", raw_fold_dir=tmp_path / "missing") == "on_demand"


def test_resolve_raw_loading_strategy_export_requires_existing_raw_fold(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _resolve_raw_loading_strategy(strategy="export", raw_fold_dir=tmp_path / "missing")
