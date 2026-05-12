from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.io import load_mne
from ..paths import WorkspacePaths, get_paths


def _load_torch_module():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "RawFoldDataset requires PyTorch. Install torch before using ds003029_eda.datasets.RawFoldDataset."
        ) from exc
    return torch


class RawFoldDataset:
    """Map-style dataset for exported raw fold tensors."""

    def __init__(self, npz_path: str | Path) -> None:
        payload = np.load(Path(npz_path), allow_pickle=False)
        torch = _load_torch_module()

        self.x_raw = torch.from_numpy(payload["x_raw"].astype(np.float32))
        self.mask = torch.from_numpy(payload["x_raw_mask"].astype(bool))
        self.y = torch.from_numpy(payload["y"].astype(np.int64))

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        return (self.x_raw[idx], self.mask[idx]), self.y[idx]


def _load_preprocessed_cache_map(artifact_root: Path) -> dict[str, Path]:
    summary_path = artifact_root / "preprocess_run_summary.csv"
    summary_df = pd.read_csv(summary_path)
    return {
        str(row.base): Path(str(row.cache_path))
        for row in summary_df.itertuples(index=False)
        if bool(row.preprocess_ok)
    }


def _pad_raw_window(window: np.ndarray, max_channels: int, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    padded = np.zeros((max_channels, n_samples), dtype=np.float32)
    mask = np.zeros((max_channels,), dtype=bool)
    n_channels = min(max_channels, window.shape[0])
    n_time = min(n_samples, window.shape[1])
    padded[:n_channels, :n_time] = window[:n_channels, :n_time].astype(np.float32, copy=False)
    mask[:n_channels] = True
    return padded, mask


class OnDemandRawFoldDataset:
    """Map-style dataset that reads raw windows directly from cached preprocessed FIF files."""

    def __init__(
        self,
        index_csv_path: str | Path,
        *,
        artifact_root: str | Path,
        max_channels: int,
        n_samples: int,
        paths: WorkspacePaths | None = None,
    ) -> None:
        self.paths = paths or get_paths()
        self.index_df = pd.read_csv(Path(index_csv_path)).reset_index(drop=True)
        self.artifact_root = Path(artifact_root)
        self.max_channels = int(max_channels)
        self.n_samples = int(n_samples)
        self.cache_map = _load_preprocessed_cache_map(self.artifact_root)
        self._torch = _load_torch_module()
        self._mne = None
        self._base_cache: dict[str, tuple[np.ndarray, float]] = {}
        self.y = self._torch.from_numpy(self.index_df["y"].to_numpy(dtype=np.int64, copy=True))

    def __len__(self) -> int:
        return int(len(self.index_df))

    def _get_mne(self):
        if self._mne is None:
            self._mne = load_mne(self.paths)
        return self._mne

    def _load_base_data(self, base: str) -> tuple[np.ndarray, float]:
        cached = self._base_cache.get(base)
        if cached is not None:
            return cached

        cache_path = self.cache_map.get(str(base))
        if cache_path is None or not cache_path.exists():
            raise FileNotFoundError(f"Missing preprocessed cache for base {base}: {cache_path}")

        raw = self._get_mne().io.read_raw_fif(cache_path, preload=True, verbose="ERROR")
        bads = set(raw.info.get("bads", []))
        good_indices = [idx for idx, channel in enumerate(raw.ch_names) if channel not in bads]
        data = raw.get_data(picks=good_indices).astype(np.float32, copy=False)
        payload = (data, float(raw.info["sfreq"]))
        self._base_cache[base] = payload
        return payload

    def __getitem__(self, idx: int):
        row = self.index_df.iloc[int(idx)]
        base = str(row.base)
        data, sfreq = self._load_base_data(base)
        start_sample = int(round(float(row.t_start_s) * sfreq))
        stop_sample = int(round(float(row.t_stop_s) * sfreq))
        window = data[:, start_sample:stop_sample]
        padded, mask = _pad_raw_window(window, max_channels=self.max_channels, n_samples=self.n_samples)
        return (
            self._torch.from_numpy(padded),
            self._torch.from_numpy(mask.astype(bool, copy=False)),
        ), self.y[int(idx)]


def infer_n_samples_from_index(index_df: pd.DataFrame, default_sfreq: float = 256.0) -> int:
    if index_df.empty or "t_start_s" not in index_df.columns or "t_stop_s" not in index_df.columns:
        return int(round(2.0 * default_sfreq))
    durations = pd.to_numeric(index_df["t_stop_s"], errors="coerce") - pd.to_numeric(index_df["t_start_s"], errors="coerce")
    duration_s = float(durations.dropna().median()) if not durations.dropna().empty else 2.0
    return int(round(duration_s * default_sfreq))


def load_fold_manifest(manifest_path: str | Path) -> dict[str, object]:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))